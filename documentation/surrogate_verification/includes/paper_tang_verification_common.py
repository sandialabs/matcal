"""
Benchmark-specific wrapper for discontinuous Tang-style surrogate verification
examples.

Most implementation lives in ``paper_surrogate_verification_common.py``.
"""
from includes.paper_surrogate_verification_common import (
    TARGET_FIELD,
    INDEPENDENT_FIELD,
    INDEPENDENT_VALUES,
    N_VALIDATION_SAMPLES,
    RANDOM_SEED,
    VALIDATION_SEED,
    INITIAL_GRID_POINTS_PER_DIMENSION,
    N_INITIAL_SAMPLES,
    TANG_BENCHMARK,
    tang_function,
    tang_python_model,
    PaperTangInitialGridVoronoiStudy,
    adaptive_history_arrays,
    collect_adaptive_surrogates_at_sample_counts,
    copy_validation_results_with_qoi_alias,
    estimate_power_law_convergence_rate,
    fit_power_law_convergence,
    get_field_score,
    make_objective,
    print_convergence_summary,
    run_fixed_validation_set,
    evaluate_scalar_surrogate_on_points,
)

from includes.paper_surrogate_verification_common import (
    make_parameters_for_benchmark,
    make_model_for_benchmark,
    make_uniform_random_samples_for_benchmark,
    run_parameter_study_for_samples as _run_parameter_study_for_samples,
    parameter_matrix_from_results_for_benchmark,
    make_2d_prediction_grid_for_benchmark,
    plot_convergence_history_for_benchmark,
    plot_function_and_surrogate_error_at_counts_for_benchmark,
    collect_training_samples_for_surrogate_count_map_for_benchmark,
    collect_training_samples_at_counts_for_benchmark,
)


BENCHMARK = TANG_BENCHMARK
PARAMETER_NAMES = list(BENCHMARK.parameter_names)
PARAMETER_BOUNDS = list(BENCHMARK.parameter_bounds)


def make_parameters():
    """
    Create MatCal parameters for the Tang-style benchmark domain.
    """
    return make_parameters_for_benchmark(BENCHMARK)


def make_model():
    """
    Create the MatCal PythonModel for the Tang-style benchmark function.
    """
    return make_model_for_benchmark(BENCHMARK)


def make_uniform_random_samples(nsamples, seed):
    """
    Generate independent uniform random samples over the Tang-style benchmark
    domain.
    """
    return make_uniform_random_samples_for_benchmark(
        nsamples,
        seed,
        BENCHMARK,
    )


def run_parameter_study_for_samples(
    parameters,
    model,
    objective,
    samples,
    working_directory,
):
    """
    Evaluate the Tang-style model at prescribed sample locations.
    """
    return _run_parameter_study_for_samples(
        parameters,
        model,
        objective,
        samples,
        working_directory,
        BENCHMARK,
    )


def parameter_matrix_from_results(results):
    """
    Extract a parameter matrix from MatCal StudyResults.
    """
    return parameter_matrix_from_results_for_benchmark(results, BENCHMARK)


def make_2d_prediction_grid(n_grid=150):
    """
    Create a two-dimensional prediction grid over the Tang-style parameter
    domain.
    """
    return make_2d_prediction_grid_for_benchmark(n_grid, BENCHMARK)


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
    Plot Tang-style verification convergence history.
    """
    return plot_convergence_history_for_benchmark(
        sample_counts,
        rmse,
        max_errors,
        method_name,
        figure_directory,
        filename,
        BENCHMARK,
        max_sample_count=max_sample_count,
    )


def plot_function_and_surrogate_error_at_counts(
    surrogates_by_count,
    training_samples_by_count,
    sample_counts_to_plot=(50, 100, 150),
    figure_directory="figures",
    filename="tang_function_and_surrogate_error_fields.png",
    method_name="surrogate",
    n_grid=150,
):
    """
    Plot true Tang-style function and absolute surrogate error fields.
    """
    return plot_function_and_surrogate_error_at_counts_for_benchmark(
        surrogates_by_count,
        training_samples_by_count,
        sample_counts_to_plot,
        figure_directory,
        filename,
        method_name,
        BENCHMARK,
        n_grid=n_grid,
    )


def collect_training_samples_for_surrogate_count_map(
    study_results,
    surrogates_by_count,
):
    """
    Collect training samples that correspond to selected adaptive surrogates.
    """
    return collect_training_samples_for_surrogate_count_map_for_benchmark(
        study_results,
        surrogates_by_count,
        BENCHMARK,
    )


def collect_training_samples_at_counts(
    study_results,
    sample_counts_to_collect=(50, 100, 150),
):
    """
    Collect training sample arrays from an adaptive study result.
    """
    return collect_training_samples_at_counts_for_benchmark(
        study_results,
        sample_counts_to_collect,
        BENCHMARK,
    )