r"""
Paper Example: KFCV-Voronoi Adaptive Sampling on the 2D Peaks Function
======================================================================

This example implements a simple benchmark from:

    Kaminsky, A. L., Wang, Y., and Pant, K.,
    "An Efficient Batch K-Fold Cross-Validation Voronoi Adaptive Sampling
    Technique for Global Surrogate Modeling,"
    Journal of Mechanical Design, 143(1), 011706, 2021.

The example uses the paper's 2D Peaks function and compares:

* KFCV-Voronoi adaptive sampling
* one-shot maximin Latin hypercube sampling, LHS

For LHS, this example only reports and plots the NRMSE convergence points.
It does not plot the LHS sample distributions.

Paper-matching setup
--------------------

Benchmark function:
    2D Peaks function, Eq. 15.

Domain:
    x1, x2 in [-5, 5].

KFCV-Voronoi initial design:
    4 by 4 uniform grid, 16 samples total.

KFCV settings:
    K = 10 folds
    l = 3 highest-error folds
    LOOCV on selected folds only
    one new sample per adaptive batch

CV metric:
    Sum of absolute physical response errors, matching the paper's

        e_i^KF = sum |y(s_j) - yhat_{S\\kf_i}(s_j)|.

Validation metric:
    NRMSE,

        NRMSE = sqrt(sum((y - yhat)^2) / sum(y^2)),

    matching Eq. 14.

Notes
-----
This example uses MatCal's current PCA/Gaussian-process surrogate machinery.
The paper used MATLAB DACE kriging. Therefore, this script is paper-faithful
with respect to the sampling method and benchmark setup, but exact numerical
curves may differ from the published figure.

This script can be computationally expensive because:

* the paper-style validation set uses 500,000 test samples;
* KFCV-Voronoi builds many cross-validation surrogates;
* LHS convergence trains one surrogate for each requested LHS sample count.

For faster debugging, reduce ``N_TEST_SAMPLES``, ``LHS_SAMPLE_COUNTS``,
``LHS_MAXIMIN_CANDIDATES``, or ``n_restarts_optimizer``.
"""

# sphinx_gallery_thumbnail_number = 1

import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import qmc
from sklearn.gaussian_process.kernels import ConstantKernel, RBF

import matcal as mc


# =============================================================================
# User options
# =============================================================================

WORKING_DIRECTORY = os.path.abspath("paper_peaks_kfcv_voronoi_vs_lhs_example")
FIGURE_DIRECTORY = os.path.join(WORKING_DIRECTORY, "figures")

TARGET_FIELD = "y"
INDEPENDENT_FIELD = "response_location"
INDEPENDENT_VALUES = np.array([0.0])

PARAMETER_NAMES = ["x1", "x2"]
PARAMETER_BOUNDS = [(-5.0, 5.0), (-5.0, 5.0)]

# Paper low-dimensional tests use a 4 by 4 initial grid for 2D functions.
INITIAL_GRID_POINTS_PER_DIMENSION = 4
N_INITIAL_SAMPLES = INITIAL_GRID_POINTS_PER_DIMENSION**2

# Final KFCV-Voronoi sample count used for convergence and sample-distribution
# plots.
FINAL_SAMPLE_COUNT = 150

# Paper uses t = 500,000 validation points. Reduce for debugging if needed.
N_TEST_SAMPLES = 1000

# KFCV-Voronoi paper settings.
K_FOLDS = 10
N_HIGH_ERROR_FOLDS = 3

# One-at-a-time Algorithm 1 sampling.
BATCH_SIZE = 1
NMAX_LOO = 1

# LHS convergence sample counts.
#
# Three points, e.g. 50, 100, 150, are not enough to see a useful convergence
# trend. This denser sequence gives a usable LHS NRMSE convergence curve while
# still keeping the total number of LHS surrogate fits manageable.
LHS_SAMPLE_COUNTS = tuple(range(20, FINAL_SAMPLE_COUNT + 1, 10))

# Number of random LHS candidates used in the simple maximin search.
# Increase for a better maximin design; decrease for faster testing.
LHS_MAXIMIN_CANDIDATES = 200

RANDOM_SEED = 54321
TEST_SEED = 12345


# =============================================================================
# Peaks benchmark function, paper Eq. 15
# =============================================================================

def peaks_function(samples):
    r"""
    Evaluate the 2D Peaks benchmark function from paper Eq. 15.

    The function is

    .. math::

        y =
        3(1-x_1)^2 e^{-x_1^2-(x_2+1)^2}
        -10\left(\frac{x_1}{5}-x_1^3-x_2^5\right)e^{-x_1^2-x_2^2}
        -\frac{1}{3} e^{-(x_1+1)^2-x_2^2},

    for :math:`x_1,x_2 \in [-5,5]`.

    :param samples: Sample locations with shape ``(n_samples, 2)``.
    :type samples: numpy.ndarray

    :return: Function values with shape ``(n_samples,)``.
    :rtype: numpy.ndarray
    """
    samples = np.asarray(samples, dtype=float)
    samples = np.atleast_2d(samples)

    x1 = samples[:, 0]
    x2 = samples[:, 1]

    term_1 = 3.0 * (1.0 - x1) ** 2 * np.exp(-(x1**2) - (x2 + 1.0) ** 2)

    term_2 = -10.0 * (
        x1 / 5.0 - x1**3 - x2**5
    ) * np.exp(-(x1**2) - x2**2)

    term_3 = -(1.0 / 3.0) * np.exp(-((x1 + 1.0) ** 2) - x2**2)

    return term_1 + term_2 + term_3


def peaks_python_model(**parameters):
    """
    MatCal PythonModel wrapper for the scalar Peaks function.

    Adaptive surrogate studies expect the modeled response to be returned as a
    response curve over an independent variable. Since the Peaks function is
    scalar-valued, this wrapper returns a one-point response curve.
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
# Paper-faithful initial-grid subclass
# =============================================================================

class PaperPeaksKFCVVoronoiStudy(mc.VoronoiAdaptiveSurrogateStudy):
    """
    Voronoi adaptive surrogate study with the paper's 4 by 4 initial grid.

    The base VoronoiAdaptiveSurrogateStudy initializes with Halton samples.
    The paper's 2D benchmark studies initialize from a 4 by 4 uniform grid.
    This subclass overrides only the initial-sampling step so that the example
    matches the paper setup.
    """

    def _make_paper_initial_grid(self):
        """
        Create the 4 by 4 uniform initial grid over [-5, 5]^2.

        :return: Initial sample locations with shape ``(16, 2)``.
        :rtype: numpy.ndarray
        """
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
        """
        Evaluate the paper's initial 4 by 4 grid, then train the first surrogate.

        :return: Tuple ``(training_params, training_data)`` used by the adaptive
            Voronoi loop.
        :rtype: tuple[numpy.ndarray, list[dict]]
        """
        initial_samples = self._make_paper_initial_grid()

        self._populate_parameter_evaluations(initial_samples)
        self._matcal_evaluate_parameter_sets_batch(self._parameter_sets_to_evaluate)

        return self._train_surrogate_with_current_results()


# =============================================================================
# Shared utility functions
# =============================================================================

def make_parameters():
    """
    Create MatCal parameters for the Peaks domain.

    :return: List of MatCal parameters.
    :rtype: list
    """
    return [
        mc.Parameter(name, lower, upper)
        for name, (lower, upper) in zip(PARAMETER_NAMES, PARAMETER_BOUNDS)
    ]


def make_model():
    """
    Create the MatCal PythonModel for the Peaks benchmark.

    :return: Configured MatCal PythonModel.
    :rtype: matcal.core.models.PythonModel
    """
    model = mc.PythonModel(peaks_python_model)
    model.set_name("paper_peaks_function_model")
    return model


def make_paper_gp_regressor_options():
    """
    Create Gaussian-process regressor options for the MatCal surrogate.

    This is not MATLAB DACE kriging, but it keeps the LHS and KFCV-Voronoi
    comparisons using the same MatCal GP surrogate settings.

    :return: Keyword options for MatCal's Gaussian-process regressor.
    :rtype: dict
    """
    return {
        "regressor_type": "Gaussian Process",
        "n_restarts_optimizer": 10,
        "alpha": 1.0e-10,
        "normalize_y": True,
    }


def run_fixed_validation_set(parameters, model, objective):
    """
    Generate the fixed validation set used for NRMSE scoring.

    The paper uses a large fixed validation set with t = 500,000 points.

    :param parameters: MatCal parameters.
    :type parameters: list

    :param model: MatCal model.
    :type model: object

    :param objective: SimulationResultsSynchronizer used by the adaptive study.
    :type objective: object

    :return: StudyResults containing fixed validation data.
    :rtype: matcal.core.study_base.StudyResults
    """
    test_study = mc.HaltonStudy(*parameters)
    test_study.add_evaluation_set(model, objective)
    test_study.set_number_of_samples(N_TEST_SAMPLES)
    test_study.set_seed(TEST_SEED)
    test_study.set_working_directory(
        os.path.join(WORKING_DIRECTORY, "fixed_validation_set"),
        remove_existing=True,
    )

    return test_study.launch()


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


def compute_nrmse(y_true, y_pred):
    r"""
    Compute the paper's NRMSE metric, Eq. 14.

    .. math::

        \mathrm{NRMSE}
        =
        \sqrt{
        \frac{\sum_i (y_i - \hat{y}_i)^2}
             {\sum_i y_i^2}
        }.

    :param y_true: Reference response values.
    :type y_true: numpy.ndarray

    :param y_pred: Surrogate response values.
    :type y_pred: numpy.ndarray

    :return: Scalar NRMSE.
    :rtype: float
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    return float(
        np.sqrt(
            np.sum((y_true - y_pred) ** 2) / np.sum(y_true**2)
        )
    )


def compute_nrmse_history(adaptive_surrogate):
    """
    Compute NRMSE for every stored KFCV-Voronoi surrogate iteration.

    This requires the study to retain every surrogate by using
    ``save_every_n_batches=1``.

    :param adaptive_surrogate: AdaptiveSurrogate object returned by the study.
    :type adaptive_surrogate: matcal.core.adaptive_surrogates.AdaptiveSurrogate

    :return: Tuple ``(sample_counts, nrmse_values)``.
    :rtype: tuple[numpy.ndarray, numpy.ndarray]
    """
    sample_counts = []
    nrmse_values = []

    y_true = adaptive_surrogate.test_responses

    for record in adaptive_surrogate.surrogate_records:
        iteration_index = record["iteration_index"]

        if iteration_index not in adaptive_surrogate.stored_surrogates:
            continue

        y_pred = adaptive_surrogate.test_predictions(
            surrogate_index=iteration_index,
        )

        sample_counts.append(record["sample_count"])
        nrmse_values.append(compute_nrmse(y_true, y_pred))

    return np.asarray(sample_counts, dtype=int), np.asarray(nrmse_values, dtype=float)


# =============================================================================
# LHS comparison utilities
# =============================================================================

def make_maximin_lhs_samples(
    nsamples,
    seed,
    n_candidates=LHS_MAXIMIN_CANDIDATES,
):
    """
    Generate a simple maximin-optimized Latin hypercube design.

    The paper compares against maximin optimized Latin hypercube sampling. This
    helper approximates that by generating several random LHS candidates and
    retaining the design with the largest minimum pairwise distance.

    :param nsamples: Number of LHS samples.
    :type nsamples: int

    :param seed: Random seed for reproducible candidate generation.
    :type seed: int

    :param n_candidates: Number of random LHS candidates to test.
    :type n_candidates: int

    :return: Scaled LHS samples with shape ``(nsamples, 2)``.
    :rtype: numpy.ndarray
    """
    rng = np.random.default_rng(seed)

    lower_bounds = np.asarray([b[0] for b in PARAMETER_BOUNDS], dtype=float)
    upper_bounds = np.asarray([b[1] for b in PARAMETER_BOUNDS], dtype=float)

    best_unit_samples = None
    best_min_distance = -np.inf

    for _ in range(n_candidates):
        candidate_seed = int(rng.integers(1, np.iinfo(np.int32).max))

        sampler = qmc.LatinHypercube(
            d=len(PARAMETER_NAMES),
            scramble=True,
            seed=candidate_seed,
        )

        unit_samples = sampler.random(n=nsamples)

        if nsamples > 1:
            min_distance = float(np.min(pdist(unit_samples)))
        else:
            min_distance = 0.0

        if min_distance > best_min_distance:
            best_min_distance = min_distance
            best_unit_samples = unit_samples

    return qmc.scale(best_unit_samples, lower_bounds, upper_bounds)


def run_parameter_study_for_samples(
    parameters,
    model,
    objective,
    samples,
    working_directory,
):
    """
    Evaluate the analytic model at a prescribed set of parameter samples.

    :param parameters: MatCal parameters.
    :type parameters: list

    :param model: MatCal model.
    :type model: object

    :param objective: MatCal objective or synchronizer.
    :type objective: object

    :param samples: Parameter samples with shape ``(n_samples, 2)``.
    :type samples: numpy.ndarray

    :param working_directory: Working directory for the parameter study.
    :type working_directory: str

    :return: Study results for the requested samples.
    :rtype: matcal.core.study_base.StudyResults
    """
    samples = np.asarray(samples, dtype=float)
    working_directory = os.path.abspath(working_directory)

    # MatCal only creates the final directory in a requested working-directory
    # path. Therefore, create the parent directory explicitly.
    parent_directory = os.path.dirname(working_directory)
    if parent_directory:
        os.makedirs(parent_directory, exist_ok=True)

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


def train_lhs_surrogate_at_count(
    parameters,
    model,
    objective,
    validation_results,
    nsamples,
    seed,
):
    """
    Train and score one maximin LHS surrogate at a specified sample count.

    :param parameters: MatCal parameters.
    :type parameters: list

    :param model: MatCal model.
    :type model: object

    :param objective: MatCal synchronizer/objective.
    :type objective: object

    :param validation_results: Fixed validation-set results.
    :type validation_results: matcal.core.study_base.StudyResults

    :param nsamples: Number of LHS training samples.
    :type nsamples: int

    :param seed: Random seed for LHS design generation.
    :type seed: int

    :return: Dictionary containing the LHS samples, study results, trained
        surrogate, and NRMSE.
    :rtype: dict
    """
    lhs_samples = make_maximin_lhs_samples(
        nsamples,
        seed=seed,
        n_candidates=LHS_MAXIMIN_CANDIDATES,
    )

    lhs_working_directory = os.path.join(
        WORKING_DIRECTORY,
        "lhs_training",
        f"n_{int(nsamples)}",
    )

    lhs_results = run_parameter_study_for_samples(
        parameters,
        model,
        objective,
        lhs_samples,
        lhs_working_directory,
    )

    gp_options = make_paper_gp_regressor_options()

    surrogate_generator = mc.SurrogateGenerator(
        lhs_results,
        interpolation_field=INDEPENDENT_FIELD,
        interpolation_locations=INDEPENDENT_VALUES,
        training_fraction=1.0,
        test_eval_info=validation_results,
        **gp_options,
    )
    surrogate_generator.set_fields_of_interest(TARGET_FIELD)
    surrogate_generator.set_PCA_details(decomp_var=0.99)

    save_name = os.path.join(
        lhs_working_directory,
        f"lhs_gp_surrogate_{int(nsamples)}",
    )

    lhs_surrogate = surrogate_generator.generate(save_name)

    validation_params = parameter_matrix_from_results(validation_results)
    true_response = peaks_function(validation_params).reshape(-1, 1)

    predicted_response = np.asarray(
        lhs_surrogate(validation_params, batch_evaluate=True)[TARGET_FIELD],
        dtype=float,
    ).reshape(true_response.shape)

    nrmse = compute_nrmse(true_response, predicted_response)

    return {
        "sample_count": int(nsamples),
        "samples": lhs_samples,
        "study_results": lhs_results,
        "surrogate": lhs_surrogate,
        "nrmse": float(nrmse),
    }


def run_lhs_comparison(
    parameters,
    model,
    objective,
    validation_results,
    sample_counts=LHS_SAMPLE_COUNTS,
):
    """
    Run maximin LHS surrogate comparisons at the requested sample counts.

    For LHS, this example only uses the results to plot NRMSE convergence
    points. It does not plot LHS sample distributions.

    :param parameters: MatCal parameters.
    :type parameters: list

    :param model: MatCal model.
    :type model: object

    :param objective: MatCal synchronizer/objective.
    :type objective: object

    :param validation_results: Fixed validation-set results.
    :type validation_results: matcal.core.study_base.StudyResults

    :param sample_counts: LHS sample counts to evaluate.
    :type sample_counts: tuple[int]

    :return: Dictionary keyed by sample count.
    :rtype: dict[int, dict]
    """
    lhs_results_by_count = {}

    for idx, sample_count in enumerate(sample_counts):
        print(f"Training maximin LHS GP surrogate with {sample_count} samples.")

        lhs_result = train_lhs_surrogate_at_count(
            parameters,
            model,
            objective,
            validation_results,
            nsamples=int(sample_count),
            seed=RANDOM_SEED + 1000 + idx,
        )

        lhs_results_by_count[int(sample_count)] = lhs_result

        print(
            f"    LHS NRMSE at {sample_count} samples: "
            f"{lhs_result['nrmse']:.6e}"
        )

    return lhs_results_by_count


# =============================================================================
# Plotting utilities
# =============================================================================

def plot_nrmse_history(
    sample_counts,
    nrmse_values,
    lhs_results_by_count=None,
):
    """
    Plot KFCV-Voronoi NRMSE history and optional LHS NRMSE convergence points.

    :param sample_counts: KFCV-Voronoi training sample counts.
    :type sample_counts: numpy.ndarray

    :param nrmse_values: KFCV-Voronoi NRMSE values.
    :type nrmse_values: numpy.ndarray

    :param lhs_results_by_count: Optional dictionary of LHS comparison results.
    :type lhs_results_by_count: dict[int, dict] or None

    :return: Matplotlib ``(fig, ax)`` pair.
    :rtype: tuple
    """
    os.makedirs(FIGURE_DIRECTORY, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.5, 5.2), constrained_layout=True)

    ax.semilogy(
        sample_counts,
        nrmse_values,
        color="tab:blue",
        linestyle="-",
        marker="o",
        markersize=4,
        linewidth=1.5,
        label="KFCV-Voronoi adaptive GP",
    )

    if lhs_results_by_count is not None:
        lhs_counts = np.asarray(sorted(lhs_results_by_count.keys()), dtype=int)
        lhs_nrmse = np.asarray(
            [lhs_results_by_count[count]["nrmse"] for count in lhs_counts],
            dtype=float,
        )

        ax.semilogy(
            lhs_counts,
            lhs_nrmse,
            color="tab:green",
            linestyle="--",
            marker="s",
            markersize=5,
            linewidth=1.8,
            label="maximin LHS one-shot GP",
        )

    ax.set_xlabel("number of training samples")
    ax.set_ylabel("NRMSE")
    ax.set_title("Peaks function: KFCV-Voronoi versus LHS")
    ax.grid(True, which="both", alpha=0.35)
    ax.legend()

    figure_path = os.path.join(
        FIGURE_DIRECTORY,
        "paper_peaks_kfcv_voronoi_vs_lhs_nrmse.png",
    )
    fig.savefig(figure_path, dpi=300)

    return fig, ax


def plot_peaks_function_with_samples_at_counts(
    study_results,
    sample_counts_to_plot=(50, 100, 150),
    n_grid=300,
):
    """
    Plot the Peaks function with KFCV-Voronoi sample locations overlaid at
    selected sample counts.

    This reproduces the style of the paper's adaptive sample-distribution plots
    for the Peaks benchmark, showing the sample locations after 50, 100, and
    150 total samples.

    :param study_results: Adaptive study training results.
    :type study_results: matcal.core.study_base.StudyResults

    :param sample_counts_to_plot: Total sample counts to show.
    :type sample_counts_to_plot: tuple[int]

    :param n_grid: Number of grid points per dimension for the function contour.
    :type n_grid: int

    :return: Matplotlib ``(fig, axes)`` pair.
    :rtype: tuple
    """
    os.makedirs(FIGURE_DIRECTORY, exist_ok=True)

    x1_samples = np.asarray(study_results.parameter_history["x1"], dtype=float)
    x2_samples = np.asarray(study_results.parameter_history["x2"], dtype=float)

    training_samples = np.column_stack((x1_samples, x2_samples))
    n_available_samples = training_samples.shape[0]

    max_requested_samples = max(sample_counts_to_plot)
    if n_available_samples < max_requested_samples:
        raise RuntimeError(
            f"Requested plot up to {max_requested_samples} samples, but only "
            f"{n_available_samples} samples are available."
        )

    x1_grid = np.linspace(PARAMETER_BOUNDS[0][0], PARAMETER_BOUNDS[0][1], n_grid)
    x2_grid = np.linspace(PARAMETER_BOUNDS[1][0], PARAMETER_BOUNDS[1][1], n_grid)

    xx, yy = np.meshgrid(x1_grid, x2_grid)
    grid_points = np.column_stack((xx.ravel(), yy.ravel()))

    zz = peaks_function(grid_points).reshape(n_grid, n_grid)

    sample_norm = plt.Normalize(vmin=1, vmax=max_requested_samples)
    sample_cmap = plt.get_cmap("viridis")

    fig, axes = plt.subplots(
        1,
        len(sample_counts_to_plot),
        figsize=(17, 5),
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )

    if len(sample_counts_to_plot) == 1:
        axes = np.asarray([axes])

    contour_handles = []

    for axis, sample_count in zip(axes, sample_counts_to_plot):
        contour = axis.contourf(
            xx,
            yy,
            zz,
            levels=40,
            cmap="coolwarm",
            alpha=0.82,
        )
        contour_handles.append(contour)

        samples_to_plot = training_samples[:sample_count]
        sample_indices = np.arange(1, sample_count + 1)

        axis.scatter(
            samples_to_plot[:, 0],
            samples_to_plot[:, 1],
            c=sample_indices,
            cmap=sample_cmap,
            norm=sample_norm,
            s=42,
            edgecolor="black",
            linewidth=0.35,
            alpha=0.95,
        )

        initial_samples = training_samples[:N_INITIAL_SAMPLES]
        axis.scatter(
            initial_samples[:, 0],
            initial_samples[:, 1],
            facecolors="none",
            edgecolors="white",
            marker="s",
            s=95,
            linewidth=1.4,
            label="initial 4x4 grid",
        )

        axis.set_title(f"KFCV-Voronoi, {sample_count} samples")
        axis.set_xlabel(r"$x_1$")
        axis.set_xlim(PARAMETER_BOUNDS[0])
        axis.set_ylim(PARAMETER_BOUNDS[1])
        axis.grid(True, color="black", alpha=0.18, linewidth=0.6)

    axes[0].set_ylabel(r"$x_2$")

    function_colorbar = fig.colorbar(
        contour_handles[-1],
        ax=axes,
        location="right",
        shrink=0.92,
        pad=0.015,
    )
    function_colorbar.set_label("Peaks function value")

    sample_mappable = plt.cm.ScalarMappable(
        norm=sample_norm,
        cmap=sample_cmap,
    )
    sample_mappable.set_array([])

    sample_colorbar = fig.colorbar(
        sample_mappable,
        ax=axes,
        location="bottom",
        shrink=0.78,
        pad=0.08,
        aspect=35,
    )
    sample_colorbar.set_label("sample index")

    fig.suptitle(
        "KFCV-Voronoi adaptive samples on the 2D Peaks function",
        fontsize=15,
    )

    figure_path = os.path.join(
        FIGURE_DIRECTORY,
        "paper_peaks_kfcv_voronoi_samples_50_100_150.png",
    )
    fig.savefig(figure_path, dpi=300)

    return fig, axes


# =============================================================================
# Main script
# =============================================================================

if __name__ == "__main__":
    os.makedirs(WORKING_DIRECTORY, exist_ok=True)
    os.makedirs(FIGURE_DIRECTORY, exist_ok=True)

    parameters = make_parameters()
    model = make_model()

    # -------------------------------------------------------------------------
    # Configure the adaptive study first so the validation set can use the same
    # internally created SimulationResultsSynchronizer.
    # -------------------------------------------------------------------------
    study = PaperPeaksKFCVVoronoiStudy(*parameters)

    study.set_independent_variable(INDEPENDENT_FIELD, INDEPENDENT_VALUES)
    study.set_target_field_name(TARGET_FIELD)
    study.add_evaluation_set(model)

    # -------------------------------------------------------------------------
    # Generate fixed validation/test set, matching the paper's use of a fixed
    # independent test set for NRMSE scoring.
    # -------------------------------------------------------------------------
    print(f"Generating fixed validation set with {N_TEST_SAMPLES} samples.")
    validation_results = run_fixed_validation_set(
        parameters,
        model,
        study.results_synchronizer,
    )

    study.set_test_data(validation_results)

    # The current VoronoiAdaptiveSurrogateStudy also passes _test_eval_info
    # directly to SurrogateGenerator during internal surrogate training.
    study._test_eval_info = validation_results

    # -------------------------------------------------------------------------
    # Paper KFCV-Voronoi settings:
    #   K = 10
    #   l = 3
    #   LOOCV refinement
    #   one-at-a-time adaptive sampling
    # -------------------------------------------------------------------------
    study.set_cross_validation_options(
        nsplits=K_FOLDS,
        nmax_folds=N_HIGH_ERROR_FOLDS,
        nmax_loo=NMAX_LOO,
        batch_size=BATCH_SIZE,
        cv_scale=1.0,
        cv_metric="sum_abs",
        group_kfold=False,
    )

    study.set_voronoi_sampling_options(
        voronoi_type="full",
        finite_only=False,
        iterative_updates=True,
        thin=None,
        random_selection=None,
    )

    voronoi_surrogate_options = make_paper_gp_regressor_options()
    voronoi_surrogate_options["decomp_var"] = 0.99
    study.set_surrogate_options(**voronoi_surrogate_options)

    # Stop by max samples, not by early error convergence.
    #
    # The base implementation stops when number_of_evaluations >
    # max_training_samples. Setting max to FINAL_SAMPLE_COUNT - 1 gives a final
    # count of FINAL_SAMPLE_COUNT for one-at-a-time sampling.
    study.set_max_training_samples(FINAL_SAMPLE_COUNT - 1)

    study.set_error_stopping_criteria(
        rmse_goal=1.0e-300,
        max_abs_error_goal=1.0e-300,
    )

    study.set_convergence_criteria(
        eps=0.0,
        convergence_metric="rmse",
    )

    # Retain every surrogate so we can reconstruct the full NRMSE history.
    study.set_surrogate_storage_options(
        best_n_surrogates=None,
        save_every_n_batches=1,
        score_metric="max_error",
    )

    study.set_seed(RANDOM_SEED)
    study.set_test_group_random_seed(TEST_SEED)

    study.set_surrogate_save_filename(
        os.path.join(
            WORKING_DIRECTORY,
            "paper_peaks_kfcv_voronoi_surrogate.joblib",
        )
    )

    study.set_working_directory(
        os.path.join(WORKING_DIRECTORY, "adaptive_training"),
        remove_existing=True,
    )

    # -------------------------------------------------------------------------
    # Run paper-style KFCV-Voronoi Peaks example.
    # -------------------------------------------------------------------------
    print("Running paper-style KFCV-Voronoi Peaks example.")
    study_results = study.launch()
    adaptive_surrogate = study.surrogate

    # -------------------------------------------------------------------------
    # Compute paper NRMSE, Eq. 14, for KFCV-Voronoi.
    # -------------------------------------------------------------------------
    sample_counts, nrmse_values = compute_nrmse_history(adaptive_surrogate)

    print("\nKFCV-Voronoi Peaks NRMSE history:")
    for n_samples, nrmse in zip(sample_counts, nrmse_values):
        print(f"  samples = {n_samples:4d}, NRMSE = {nrmse:.6e}")

    print("\nKFCV-Voronoi final result:")
    print(f"  final sample count: {sample_counts[-1]}")
    print(f"  final NRMSE:        {nrmse_values[-1]:.6e}")

    # -------------------------------------------------------------------------
    # Run maximin LHS one-shot comparison at enough sample counts to give a
    # usable convergence curve.
    # -------------------------------------------------------------------------

    # lhs_results_by_count = run_lhs_comparison(
    #     parameters,
    #     model,
    #     study.results_synchronizer,
    #     validation_results,
    #     sample_counts=LHS_SAMPLE_COUNTS,
    # )

    # print("\nLHS comparison NRMSE:")
    # for count in sorted(lhs_results_by_count):
    #     print(
    #         f"  LHS samples = {count:4d}, "
    #         f"NRMSE = {lhs_results_by_count[count]['nrmse']:.6e}"
    #     )

    # -------------------------------------------------------------------------
    # Plots.
    # -------------------------------------------------------------------------
    plot_nrmse_history(
        sample_counts,
        nrmse_values,
#        lhs_results_by_count=lhs_results_by_count,
    )

    plot_peaks_function_with_samples_at_counts(
        study_results,
        sample_counts_to_plot=(50, 100, 150),
        n_grid=300,
    )

    print(f"\nFigures saved to: {FIGURE_DIRECTORY}")
    plt.show()