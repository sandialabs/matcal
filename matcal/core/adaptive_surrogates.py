"""
This module contains adaptive surrogates. 
"""
from collections import OrderedDict
import copy
import numpy as np
import os

from matcal.core.logger import initialize_matcal_logger
from matcal.core.objective import SimulationResultsSynchronizer
from matcal.core.parameter_studies import HaltonStudy
from matcal.core.qoi_extractor import UserDefinedExtractor
from matcal.core.state import State
from matcal.core.study_base import StudyResults
from matcal.core.utilities import (check_value_is_positive_integer, 
                                   check_value_is_positive_integer_or_none,
                                   check_value_is_nonempty_str, 
                                   check_value_is_array_like_of_reals, 
                                   check_value_is_positive_real, 
                                   check_value_is_bool, 
                                   check_value_is_nonnegative_integer, 
                                   check_item_is_correct_type)
from matcal.core.serializer_wrapper import matcal_load, matcal_save
from matcal.core.surrogates import (_root_mean_squared_error, 
                                    _max_error_inf_norm, 
                                    _process_surrogate_args_call, 
                                    _check_params_in_range, 
                                    _convert_param_array_to_dict, 
                                    _global_r2_score)

logger = initialize_matcal_logger(__name__)


def _get_or_create_matplotlib_axes(figure=None, axes=None):
    if axes is not None:
        return axes.figure, axes

    import matplotlib.pyplot as plt

    if figure is None:
        return plt.subplots()

    if len(figure.axes) > 0:
        return figure, figure.axes[0]

    return figure, figure.add_subplot(1, 1, 1)


def _replace_underscores(text):
    if text is None:
        return None
    return str(text).replace("_", " ")


def _format_axis_label(label, units=None):
    label = _replace_underscores(label)
    if label is None:
        return None

    if units is None or str(units).strip() == "":
        return label

    return f"{label} ({units})"


def _merge_plot_style(default_style, user_style):
    style = default_style.copy()
    if user_style is not None:
        style.update(user_style)
    return style


def _apply_axis_limits(axes, xlim=None, ylim=None):
    if xlim is not None:
        axes.set_xlim(xlim)
    if ylim is not None:
        axes.set_ylim(ylim)


def _apply_axis_scales(axes, xscale=None, yscale=None):
    if xscale is not None:
        axes.set_xscale(xscale)
    if yscale is not None:
        axes.set_yscale(yscale)


def _apply_axis_labels(axes, xlabel=None, ylabel=None, title=None):
    if xlabel is not None:
        axes.set_xlabel(xlabel)
    if ylabel is not None:
        axes.set_ylabel(ylabel)
    if title is not None:
        axes.set_title(title)


def _apply_grid(axes, grid):
    if grid is not None:
        axes.grid(grid)


def _apply_legend(axes, show_legend):
    if show_legend:
        axes.legend()


def _apply_axes_plot_options(
    axes,
    xlabel=None,
    ylabel=None,
    title=None,
    xlim=None,
    ylim=None,
    xscale=None,
    yscale=None,
    grid=None,
):
    _apply_axis_labels(axes, xlabel, ylabel, title)
    _apply_axis_limits(axes, xlim, ylim)
    _apply_axis_scales(axes, xscale, yscale)
    _apply_grid(axes, grid)


def _as_2d_response_array(values):
    values = np.asarray(values)
    if values.ndim == 1:
        return values.reshape(-1, 1)
    return values


def _validate_sample_indices(sample_indices, n_samples):
    indices = np.atleast_1d(np.asarray(sample_indices, dtype=int))

    if np.any(indices < 0) or np.any(indices >= n_samples):
        raise ValueError(
            "sample_indices contains an index outside the valid range "
            f"[0, {n_samples - 1}]."
        )

    return indices


def _default_sample_indices(n_samples):
    return np.arange(n_samples, dtype=int)


def _get_sample_indices(sample_indices, n_samples):
    if sample_indices is None:
        return _default_sample_indices(n_samples)
    return _validate_sample_indices(sample_indices, n_samples)


def _default_test_data_style():
    return {
        "color": "black",
        "linestyle": "None",
        "marker": "o",
        "alpha": 0.55,
        "markersize": 4,
    }


def _default_surrogate_style():
    return {
        "color": "tab:blue",
        "linestyle": "-",
        "marker": None,
        "alpha": 0.85,
        "linewidth": 1.5,
    }


def _default_surrogate_error_style():
    return {
        "color": "tab:red",
        "linestyle": "-",
        "marker": None,
        "alpha": 0.9,
        "linewidth": 1.8,
    }


def _default_error_history_styles():
    return {
        "rmse": {
            "color": "tab:blue",
            "linestyle": "-",
            "marker": "o",
            "label": "RMSE",
        },
        "max_error": {
            "color": "tab:orange",
            "linestyle": "-",
            "marker": "s",
            "label": "max absolute error",
        },
        "r2": {
            "color": "tab:green",
            "linestyle": "-",
            "marker": "^",
            "label": r"$R^2$",
        },
        "score": {
            "color": "tab:green",
            "linestyle": "-",
            "marker": "^",
            "label": r"$R^2$",
        },
    }


def _add_label_to_first_curve(style, label, curve_index):
    style = style.copy()
    if curve_index == 0:
        style.setdefault("label", label)
    else:
        style.setdefault("label", "_nolegend_")
    return style


def _validate_error_type(error_type):
    check_value_is_nonempty_str(error_type, "error_type")
    error_type = error_type.lower().strip()

    valid_error_types = ("absolute", "signed", "squared")
    if error_type not in valid_error_types:
        raise ValueError(
            f"error_type must be one of {valid_error_types}. "
            f"Received '{error_type}'."
        )

    return error_type


def _validate_error_statistic(error_statistic):
    if error_statistic is None:
        return None

    check_value_is_nonempty_str(error_statistic, "error_statistic")
    error_statistic = error_statistic.lower().strip()

    valid_error_statistics = ("mean", "median", "max")
    if error_statistic not in valid_error_statistics:
        raise ValueError(
            f"error_statistic must be one of {valid_error_statistics} or None. "
            f"Received '{error_statistic}'."
        )

    return error_statistic


def _error_units_for_type(error_type, target_field_units, error_units):
    if error_units is not None:
        return error_units

    if error_type == "squared" and target_field_units is not None:
        return f"{target_field_units}^2"

    return target_field_units


def _get_parameter_bounds(parameters):
    param_bounds = []
    for name, parameter in parameters.items():
        param_bounds.append(np.atleast_2d([parameter.get_lower_bound(),
                                          parameter.get_upper_bound()]))
    bounds = np.r_[*param_bounds]
    return bounds


def _validate_cv_scale(cv_scale):
    """
    Validate and normalize the cross-validation response scaling option.

    Accepted values are ``None``, a positive scalar, a positive array-like
    object, or the string ``"cbrt"``.
    """
    if cv_scale is None:
        return None

    if isinstance(cv_scale, str):
        cv_scale = cv_scale.lower().strip()
        if cv_scale == "cbrt":
            return cv_scale
        raise ValueError("cv_scale string option must be 'cbrt'.")

    scale = np.asarray(cv_scale, dtype=float)
    if scale.size == 0 or np.any(~np.isfinite(scale)) or np.any(scale <= 0):
        raise ValueError("cv_scale must contain finite positive values.")

    return float(scale) if scale.ndim == 0 else scale


def _get_valid_kfold_split_count(nsplits, n_samples):
    """
    Return a valid number of K-fold splits for the current sample count.

    If the requested split count is larger than the sample count, it is reduced
    to a valid value and a warning is emitted.
    """
    check_value_is_nonnegative_integer(nsplits, "nsplits")

    if nsplits == 0:
        return 0

    if nsplits == 1:
        raise ValueError("nsplits must be 0 to disable K-fold CV or at least 2.")


    if n_samples < 2:
        raise ValueError("At least two samples are required for K-fold CV.")

    if nsplits <= n_samples:
        return int(nsplits)

    new_nsplits = min(max(2, n_samples // 2), n_samples)
    logger.warning(f"Reducing nsplits from {nsplits} to {new_nsplits}.")
    return int(new_nsplits)


def _setup_pyapprox_adaptive_sparse_grid_fitter(
    n_parameters: int,
    n_qois: int,
    bounds,
    basis_type: str = "lagrange",
    piecewise_degree: int = 2,
    max_level: int = 20,
    pnorm: float = 1.0,
):
    """
    Build a PyApprox adaptive sparse-grid fitter.

    Notes
    -----
    * Parameters are assumed to be in native/physical parameter space.
    * The physical parameter bounds are supplied to PyApprox through the
      one-dimensional marginal distributions.
    * basis_type:
        - 'lagrange': global Clenshaw-Curtis Lagrange
        - 'piecewise': local piecewise polynomial basis
    """
    from pyapprox.util.backends.numpy import NumpyBkd
    from pyapprox.probability.univariate.uniform import UniformMarginal

    from pyapprox.surrogates.sparsegrids import (
        SingleFidelityAdaptiveSparseGridFitter,
        TensorProductSubspaceFactory,
    )
    from pyapprox.surrogates.sparsegrids.basis_factory import (
        ClenshawCurtisLagrangeFactory,
        PiecewiseFactory,
    )
    from pyapprox.surrogates.affine.indices import (
        ClenshawCurtisGrowthRule,
        CubicNestedGrowthRule,
        MaxLevelCriteria,
    )
    from pyapprox.surrogates.sparsegrids.error_indicators import (
        VarianceChangeIndicator,
    )

    bkd = NumpyBkd()

    bounds = np.asarray(bounds, dtype=float)
    expected_shape = (n_parameters, 2)
    if bounds.shape != expected_shape:
        raise ValueError(
            f"bounds must have shape {expected_shape}. Received {bounds.shape}."
        )

    if np.any(bounds[:, 1] <= bounds[:, 0]):
        raise ValueError(
            "Each parameter upper bound must be greater than its lower bound."
        )

    marginals = [
        UniformMarginal(float(bounds[ii, 0]), float(bounds[ii, 1]), bkd)
        for ii in range(n_parameters)
    ]

    basis_type = basis_type.lower().strip()
    if basis_type not in ("lagrange", "piecewise"):
        raise ValueError(
            f"basis_type must be 'lagrange' or 'piecewise'. Got '{basis_type}'."
        )

    if basis_type == "lagrange":
        factories = [
            ClenshawCurtisLagrangeFactory(marginals[ii], bkd)
            for ii in range(n_parameters)
        ]
        growth = ClenshawCurtisGrowthRule()
    else:
        if piecewise_degree not in (1, 2, 3):
            raise ValueError("piecewise_degree must be 1, 2, or 3")

        poly_type = {1: "linear", 2: "quadratic", 3: "cubic"}[piecewise_degree]
        factories = [
            PiecewiseFactory(marginals[ii], bkd, poly_type=poly_type)
            for ii in range(n_parameters)
        ]
        growth = CubicNestedGrowthRule() if poly_type == "cubic" else ClenshawCurtisGrowthRule()

    tp_factory = TensorProductSubspaceFactory(bkd, factories, growth)
    admissibility = MaxLevelCriteria(max_level=max_level, pnorm=pnorm, bkd=bkd)
    error_indicator = VarianceChangeIndicator(bkd)

    fitter = SingleFidelityAdaptiveSparseGridFitter(
        bkd,
        tp_factory,
        admissibility,
        error_indicator=error_indicator,
    )
    return fitter


class AdaptiveSurrogate:
    """
    Stores retained surrogate objects, training/test score histories, test data,
    and metadata describing the progress of adaptive surrogate training.

    The adaptive surrogate study may generate many surrogate models during
    training. To avoid unnecessarily large ``.joblib`` files, this class does
    not have to retain every trained surrogate object. Instead, the retention
    policy is configurable through
    :meth:`AdaptiveSurrogateStudyBase.set_surrogate_storage_options`.

    Regardless of the surrogate-object retention policy, this class always
    stores:

    * the test parameters used to score each surrogate,
    * the test responses used to score each surrogate,
    * the RMSE history,
    * the maximum absolute error history,
    * the :math:`R^2` score history,
    * the sample-count history, and
    * one metadata record for every adaptive-training batch.

    Retained surrogate objects are stored in ``stored_surrogates`` and are
    keyed by the adaptive-training iteration index. The corresponding scores
    and metadata for retained surrogates are available through
    ``stored_surrogate_scores`` using the same keys.

    By default, only the best surrogate, as measured by maximum absolute error
    on the test set, is retained.
    """

    _VALID_STORAGE_METRICS = ("rmse", "max_error", "r2", "score")

    def __init__(self, target_field_name, indep_variable_name,
                indep_variable_values, test_params, test_responses,
                param_names, bounds,
                storage_best_n_surrogates=1,
                storage_every_n_batches=None,
                storage_score_metric="max_error"):
        """
        Create an :class:`AdaptiveSurrogate` instance.

        :param str target_field_name: Name of the model field that the surrogate
            approximates, e.g. ``"temperature"``, ``"load"``, or ``"f"``.

        :param str indep_variable_name: Name of the auxiliary independent
            variable attached to the surrogate output, e.g. ``"time"``,
            ``"x"``, or ``"x_position"``. This is not a model input parameter;
            it is the independent coordinate for the predicted response.

        :param indep_variable_values: Values of the independent variable at
            which the surrogate response is reported.
        :type indep_variable_values: array-like of real numbers

        :param test_params: Parameter samples used to evaluate surrogate
            accuracy. These are always stored, regardless of the surrogate
            retention policy.
        :type test_params: :class:`numpy.ndarray`

        :param test_responses: Model responses corresponding to
            ``test_params``. These are always stored, regardless of the surrogate
            retention policy.
        :type test_responses: :class:`numpy.ndarray` of shape
            ``(n_test_samples, n_qois)``

        :param param_names: Ordered parameter names. This order defines the
            mapping between positional arguments and model parameters when the
            surrogate is called.
        :type param_names: list[str]

        :param bounds: Physical parameter bounds. Expected shape is
            ``(n_parameters, 2)``, where column 0 is the lower bound and column
            1 is the upper bound.
        :type bounds: :class:`numpy.ndarray`

        :param storage_best_n_surrogates: Number of best surrogate objects to
            retain according to ``storage_score_metric``. If ``None``,
            score-based retention is disabled.
        :type storage_best_n_surrogates: int or None

        :param storage_every_n_batches: If provided, retain every N-th adaptive
            batch surrogate in addition to any score-based retained surrogates.
        :type storage_every_n_batches: int or None

        :param storage_score_metric: Metric used to identify the best
            surrogates. Supported values are ``"rmse"``, ``"max_error"``,
            ``"r2"``, and ``"score"``. ``"score"`` is an alias for ``"r2"``.
        :type storage_score_metric: str
        """
        self._surrogates = OrderedDict()
        self._surrogate_iteration_records = []

        self._root_mean_squared_errors: list[float] = []
        self._max_errors: list[float] = []
        self._r2_scores: list[float] = []
        self._sample_counts: list[int] = []

        self._target_field_name: str = target_field_name
        self._indep_variable_name = indep_variable_name
        self._indep_variable_values = np.asarray(indep_variable_values)

        # Always persisted. These are needed to understand/diagnose all scores.
        self._test_params = test_params
        self._test_responses = test_responses

        self._param_names = param_names
        self._bounds = bounds
        self._enforce_training_data_parameter_range = True

        self._storage_best_n_surrogates = None
        self._storage_every_n_batches = None
        self._storage_score_metric = None
        self.set_surrogate_storage_options(
            best_n_surrogates=storage_best_n_surrogates,
            save_every_n_batches=storage_every_n_batches,
            score_metric=storage_score_metric,
        )

    def set_surrogate_storage_options(self, best_n_surrogates=1,
                                      save_every_n_batches=None,
                                      score_metric="max_error"):
        """
        Configure which surrogate model objects are retained.

        This method controls only the storage of trained surrogate objects.
        All score histories, test parameters, test responses, sample counts,
        and per-batch metadata records are always retained.

        The retention policy can combine two independent rules:

        * retain the best ``N`` surrogate objects according to a score metric;
        * retain every ``N``-th adaptive-training batch surrogate.

        If both rules are active, the retained surrogate set is the union of
        the best-score surrogates and the periodic-batch surrogates.

        :param best_n_surrogates: Retain the best N surrogate objects according
            to ``score_metric``. If ``None``, score-based retention is disabled.
        :type best_n_surrogates: int or None

        :param save_every_n_batches: Retain every N-th adaptive batch surrogate.
            If ``None``, periodic retention is disabled.
        :type save_every_n_batches: int or None

        :param score_metric: Metric used to rank the best surrogates. Supported
            values are ``"rmse"``, ``"max_error"``, ``"r2"``, and ``"score"``.
            ``"score"`` is treated as an alias for ``"r2"``.
        :type score_metric: str

        :raises TypeError: If ``best_n_surrogates`` or
            ``save_every_n_batches`` is not ``None`` and is not an integer, or
            if ``score_metric`` is not a string.

        :raises ValueError: If either storage count is non-positive, if
            ``score_metric`` is not supported, or if both retention rules are
            disabled.

        **Examples**

        Retain only the best surrogate by maximum absolute error:

        >>> study.set_surrogate_storage_options(best_n_surrogates=1)

        Retain the best five surrogates by RMSE:

        >>> study.set_surrogate_storage_options(
        ...     best_n_surrogates=5,
        ...     score_metric="rmse",
        ... )

        Retain every tenth adaptive batch surrogate:

        >>> study.set_surrogate_storage_options(
        ...     best_n_surrogates=None,
        ...     save_every_n_batches=10,
        ... )

        Retain the best two surrogates and every fifth batch surrogate by RMSE:

        >>> study.set_surrogate_storage_options(
        ...     best_n_surrogates=2,
        ...     save_every_n_batches=5,
        ...     score_metric="rmse",
        ... )
        """
        check_value_is_positive_integer_or_none(best_n_surrogates, "best_n_surrogates")
        check_value_is_positive_integer_or_none(save_every_n_batches, "save_every_n_batches")
        check_value_is_nonempty_str(score_metric, "score_metric")

        score_metric = score_metric.lower().strip()
        if score_metric not in self._VALID_STORAGE_METRICS:
            raise ValueError(
                "Invalid surrogate storage score metric. "
                f"Supported metrics are {self._VALID_STORAGE_METRICS}. "
                f"Received '{score_metric}'."
            )

        if best_n_surrogates is None and save_every_n_batches is None:
            raise ValueError(
                "At least one surrogate retention option must be active. "
                "Set best_n_surrogates to a positive integer or "
                "save_every_n_batches to a positive integer."
            )

        if score_metric == "score":
            score_metric = "r2"

        self._storage_best_n_surrogates = best_n_surrogates
        self._storage_every_n_batches = save_every_n_batches
        self._storage_score_metric = score_metric

    def enforce_training_data_parameter_range(self, enforce_training_data_parameter_range=True):
        """
        Activate or deactivate parameter-range enforcement during surrogate calls.

        By default, the surrogate raises an error if it is evaluated at
        parameter values outside the parameter bounds used to generate the
        training data. Calling this method with ``False`` permits extrapolative
        surrogate calls outside the training-data parameter range. Calling it
        again with ``True`` restores the default range-checking behavior.

        :param enforce_training_data_parameter_range: If ``True``, reject
            surrogate calls outside the training parameter bounds. If ``False``,
            allow calls outside the training parameter bounds.
        :type enforce_training_data_parameter_range: bool

        :raises TypeError: If ``enforce_training_data_parameter_range`` is not
            a boolean.
        """
        check_value_is_bool(enforce_training_data_parameter_range, 
                            "enforce_training_data_parameter_range")
        self._enforce_training_data_parameter_range = enforce_training_data_parameter_range

    def _add_iteration(self, surrogate, nsamples) -> None:
        """
        Add one adaptive-training iteration.

        The candidate surrogate is scored immediately. It is retained only if it
        satisfies the configured storage policy.
        """
        surrogate_values = self._evaluate_surrogate_object(
            surrogate, self._test_params, batch_evaluate=True
        )[self._target_field_name]

        rmse = _root_mean_squared_error(self._test_responses, surrogate_values)
        max_abs_error = _max_error_inf_norm(self._test_responses, surrogate_values)
        score = _global_r2_score(self._test_responses, surrogate_values)

        self._root_mean_squared_errors.append(rmse)
        self._max_errors.append(max_abs_error)
        self._r2_scores.append(score)
        self._sample_counts.append(nsamples)

        iteration_index = len(self._surrogate_iteration_records)
        record = OrderedDict()
        record["iteration_index"] = iteration_index
        record["batch_number"] = iteration_index + 1
        record["sample_count"] = nsamples
        record["rmse"] = float(rmse)
        record["max_error"] = float(max_abs_error)
        record["r2"] = float(score) if not np.isnan(score) else np.nan
        record["surrogate_stored"] = False
        record["storage_reason"] = []

        self._surrogate_iteration_records.append(record)
        self._update_retained_surrogates(surrogate, record)

    def _evaluate_surrogate_object(self, surrogate, *args, batch_evaluate=False,
                                   transpose=False, **kwargs):
        """
        Evaluate a candidate or retained surrogate object.

        Subclasses can override this for surrogate libraries with special call
        signatures.
        """
        return surrogate(*args, batch_evaluate=batch_evaluate,
                         transpose=transpose, **kwargs)

    def _update_retained_surrogates(self, surrogate, record):
        iteration_index = record["iteration_index"]

        retain_now = False

        if self._storage_best_n_surrogates is not None:
            retain_now = True
            record["storage_reason"].append("best_candidate")

        if self._storage_every_n_batches is not None:
            if record["batch_number"] % self._storage_every_n_batches == 0:
                retain_now = True
                record["storage_reason"].append("periodic")

        if retain_now:
            self._surrogates[iteration_index] = copy.deepcopy(surrogate)
            record["surrogate_stored"] = True

        self._prune_score_based_retained_surrogates()

    def _metric_value_for_record(self, record):
        metric = self._storage_score_metric
        value = record[metric]
        if value is None or np.isnan(value):
            if metric == "r2":
                return -np.inf
            return np.inf
        return value

    def _sorted_record_indices_by_metric(self):
        metric = self._storage_score_metric
        reverse = metric == "r2"

        return [
            rec["iteration_index"]
            for rec in sorted(
                self._surrogate_iteration_records,
                key=self._metric_value_for_record,
                reverse=reverse,
            )
        ]

    def _prune_score_based_retained_surrogates(self):
        """
        Retain:
          * all periodic-retention surrogates, and
          * the best N score-based surrogates.

        If only score-based retention is active, this leaves only the best N.
        """
        if self._storage_best_n_surrogates is None:
            return

        best_indices = set(
            self._sorted_record_indices_by_metric()[:self._storage_best_n_surrogates]
        )

        periodic_indices = set()
        for rec in self._surrogate_iteration_records:
            if "periodic" in rec["storage_reason"]:
                periodic_indices.add(rec["iteration_index"])

        keep_indices = best_indices | periodic_indices

        for idx in list(self._surrogates.keys()):
            if idx not in keep_indices:
                del self._surrogates[idx]

        for rec in self._surrogate_iteration_records:
            idx = rec["iteration_index"]
            rec["surrogate_stored"] = idx in self._surrogates
            reasons = []
            if idx in best_indices:
                reasons.append("best")
            if idx in periodic_indices:
                reasons.append("periodic")
            rec["storage_reason"] = reasons

    def _select_surrogate(self, surrogate_index=-1):
        if len(self._surrogates) == 0:
            raise RuntimeError("No surrogate objects are currently stored.")

        if surrogate_index == "best":
            idx = self.best_surrogate_iteration_index
            return self._surrogates[idx]

        if surrogate_index == "latest":
            idx = max(self._surrogates.keys())
            return self._surrogates[idx]

        if not isinstance(surrogate_index, int):
            raise TypeError(
                "surrogate_index must be an integer, 'best', or 'latest'."
            )

        # Prefer exact iteration-index lookup for nonnegative indices.
        if surrogate_index >= 0 and surrogate_index in self._surrogates:
            return self._surrogates[surrogate_index]

        # Otherwise treat as positional index into retained surrogates.
        keys = list(self._surrogates.keys())
        try:
            return self._surrogates[keys[surrogate_index]]
        except IndexError:
            raise IndexError(
                f"Retained surrogate index {surrogate_index} is invalid. "
                f"Retained iteration indices are {keys}."
            )

    @property
    def current_surrogate(self):
        """
        Return the most recent retained surrogate object.

        This is not necessarily the most recently trained surrogate. If the
        storage policy retains only the best surrogate, then a newly trained
        surrogate that does not improve the selected metric may be discarded.

        :return: Most recent retained surrogate, or ``None`` if no surrogate has
            been retained.
        :rtype: object or None
        """
        if not self._surrogates:
            return None
        latest_idx = max(self._surrogates.keys())
        return self._surrogates[latest_idx]

    @property
    def best_surrogate(self):
        """
        Return the best retained surrogate according to the storage metric.

        :return: Best retained surrogate, or ``None`` if no surrogate has been
            retained.
        :rtype: object or None
        """
        if not self._surrogates:
            return None
        return self._surrogates[self.best_surrogate_iteration_index]

    @property
    def best_surrogate_iteration_index(self):
        """
        Return the adaptive-training iteration index of the best retained surrogate.

        :return: Iteration index for the best retained surrogate, or ``None`` if
            no surrogate has been retained.
        :rtype: int or None
        """
        if not self._surrogate_iteration_records:
            return None
        best_order = self._sorted_record_indices_by_metric()
        for idx in best_order:
            if idx in self._surrogates:
                return idx
        return None

    @property
    def stored_surrogates(self):
        """
        Return the retained surrogate objects.

        The returned object maps adaptive-training iteration index to surrogate
        object. These keys can be used to retrieve the corresponding score
        records from :attr:`stored_surrogate_scores`.

        :return: Retained surrogate objects keyed by iteration index.
        :rtype: OrderedDict[int, object]
        """
        return self._surrogates

    @property
    def surrogate_records(self):
        """
        Return metadata records for all adaptive-training batches.

        This includes records for batches whose surrogate objects were retained
        and batches whose surrogate objects were discarded. Each record contains:

        * ``iteration_index``
        * ``batch_number``
        * ``sample_count``
        * ``rmse``
        * ``max_error``
        * ``r2``
        * ``surrogate_stored``
        * ``storage_reason``

        :return: Per-batch surrogate metadata records.
        :rtype: list[OrderedDict]
        """
        return self._surrogate_iteration_records

    @property
    def stored_surrogate_scores(self):
        """
        Return score records for retained surrogate objects.

        This provides a clean link between retained surrogate objects and their
        scores. The keys match the keys in :attr:`stored_surrogates`.

        :return: Score records for retained surrogates keyed by iteration index.
        :rtype: OrderedDict[int, OrderedDict]
        """
        records = OrderedDict()
        for idx in self._surrogates:
            records[idx] = self._surrogate_iteration_records[idx]
        return records

    @property
    def test_params(self):
        """
        Return the test parameters used to score every adaptive surrogate.

        These values are always stored regardless of how many surrogate objects
        are retained.

        :return: Test parameter samples.
        :rtype: numpy.ndarray
        """
        return self._test_params

    @property
    def test_responses(self):
        """
        Return the test responses used to score every adaptive surrogate.

        These values are always stored regardless of how many surrogate objects
        are retained.

        :return: Test response values.
        :rtype: numpy.ndarray
        """
        return self._test_responses

    def test_predictions(self, surrogate_index="best"):
        """
        Return retained-surrogate predictions at the stored test-parameter
        locations.

        :param surrogate_index: Retained surrogate selector. Supported values
            are ``"best"``, ``"latest"``, a retained adaptive iteration index,
            or a positional retained-surrogate index. Defaults to ``"best"``.
        :type surrogate_index: int or str

        :return: Surrogate predictions with shape ``(n_test_samples, n_qois)``.
        :rtype: numpy.ndarray
        """
        return self._get_test_prediction_array(surrogate_index)

    def test_errors(self, surrogate_index="best", error_type="signed"):
        """
        Return surrogate errors at the stored test-parameter locations.

        The raw signed error is

        .. math::

            e = \\hat{y} - y

        where ``hat(y)`` is the surrogate prediction and ``y`` is the stored
        test response.

        Supported error definitions are:

        * ``"signed"``: ``surrogate - test``
        * ``"absolute"``: ``abs(surrogate - test)``
        * ``"squared"``: ``(surrogate - test)**2``

        :param surrogate_index: Retained surrogate selector. Defaults to
            ``"best"``.
        :type surrogate_index: int or str

        :param error_type: Error definition. Must be ``"signed"``,
            ``"absolute"``, or ``"squared"``.
        :type error_type: str

        :return: Error array with shape ``(n_test_samples, n_qois)``.
        :rtype: numpy.ndarray
        """
        error_type = _validate_error_type(error_type)
        return self._get_surrogate_error_array(surrogate_index, error_type)

    def _validate_test_sample_error_metric(self, metric):
        check_value_is_nonempty_str(metric, "metric")
        metric = metric.lower().strip()

        valid_metrics = (
            "max_error",
            "max_abs_error",
            "linf",
            "rmse",
            "mae",
            "mean_abs_error",
        )
        if metric not in valid_metrics:
            raise ValueError(
                "Unsupported test-sample error metric. Supported values are "
                "'max_error', 'max_abs_error', 'linf', 'rmse', 'mae', and "
                "'mean_abs_error'. "
                f"Received '{metric}'."
            )

        return metric

    def test_sample_errors(self, surrogate_index="best", metric="max_error"):
        """
        Return one scalar error per stored test sample.

        Supported metrics are:

        * ``"max_error"``, ``"max_abs_error"``, or ``"linf"``:
          maximum absolute error over the surrogate independent variable;
        * ``"rmse"``:
          root-mean-squared error over the surrogate independent variable;
        * ``"mae"`` or ``"mean_abs_error"``:
          mean absolute error over the surrogate independent variable.

        :param surrogate_index: Retained surrogate selector. Defaults to
            ``"best"``.
        :type surrogate_index: int or str

        :param metric: Per-sample error metric.
        :type metric: str

        :return: One scalar error value per stored test sample.
        :rtype: numpy.ndarray
        """
        metric = self._validate_test_sample_error_metric(metric)
        raw_error = self._raw_surrogate_error(surrogate_index)

        if metric in ("max_error", "max_abs_error", "linf"):
            return np.nanmax(np.abs(raw_error), axis=1)

        if metric == "rmse":
            return np.sqrt(np.nanmean(raw_error**2, axis=1))

        if metric in ("mae", "mean_abs_error"):
            return np.nanmean(np.abs(raw_error), axis=1)

    def worst_test_sample_indices(self, n=5, surrogate_index="best",
                                  metric="max_error"):
        """
        Return the indices of the worst N stored test samples.

        The samples are ranked using :meth:`test_sample_errors` and returned
        from largest error to smallest error.

        :param n: Number of worst test samples to return.
        :type n: int

        :param surrogate_index: Retained surrogate selector. Defaults to
            ``"best"``.
        :type surrogate_index: int or str

        :param metric: Per-sample error metric used for ranking. Supported
            values are ``"max_error"``, ``"max_abs_error"``, ``"linf"``,
            ``"rmse"``, ``"mae"``, and ``"mean_abs_error"``.
        :type metric: str

        :return: Test-sample indices sorted from worst to best.
        :rtype: numpy.ndarray
        """
        check_value_is_positive_integer(n, "n")

        sample_errors = self.test_sample_errors(
            surrogate_index=surrogate_index,
            metric=metric,
        )
        n = min(n, sample_errors.size)
        return np.argsort(sample_errors)[::-1][:n]

    @property
    def rmse_history(self):
        """
        Return the full root-mean-squared-error history.

        The RMSE is calculated for each adaptive-training batch using the stored
        test responses and the candidate surrogate predictions:

        .. math::

            \\mathrm{RMSE}
            =
            \\sqrt{
            \\frac{1}{N_{\\text{samples}}N_{\\text{qoi}}}
            \\sum_{i=1}^{N_{\\text{samples}}}
            \\sum_{j=1}^{N_{\\text{qoi}}}
            \\left(
            R_{\\text{test},ij} - \\hat{R}_{ij}
            \\right)^2
            }

        where :math:`R_{\\text{test}}` is the test response and
        :math:`\\hat{R}` is the surrogate response.

        :return: RMSE value for every adaptive-training batch.
        :rtype: list[float]
        """
        return self._root_mean_squared_errors

    @property
    def max_error_history(self):
        """
        Return the full maximum absolute error history.

        The maximum absolute error is calculated as

        .. math::

            E_{\\max}
            =
            \\lVert
            \\mathbf{R}_{\\text{test}} - \\hat{\\mathbf{R}}
            \\rVert_{\\infty}

        where :math:`\\mathbf{R}_{\\text{test}}` is the test response and
        :math:`\\hat{\\mathbf{R}}` is the surrogate response.

        :return: Maximum absolute error for every adaptive-training batch.
        :rtype: list[float]
        """
        return self._max_errors

    def score(self, surrogate_index=-1):
        """
        Return the :math:`R^2` test score for an adaptive-training batch.

        The score history is retained for all batches, even if the corresponding
        surrogate object was discarded by the storage policy.

        :param surrogate_index: Index into the full score history. The default
            ``-1`` returns the score from the most recent adaptive-training
            batch.
        :type surrogate_index: int

        :return: :math:`R^2` score for the selected batch. Returns ``nan`` when
            the score is not defined, such as for a single scalar QoI.
        :rtype: float
        """        
        return self._r2_scores[surrogate_index]

    @property
    def sample_count_history(self):
        """
        Return the number of training samples used at each adaptive batch.

        :return: Training sample count for every adaptive-training batch.
        :rtype: list[int]
        """
        return self._sample_counts
    
    def __call__(self, *args, surrogate_index="best", batch_evaluate=False,
                 transpose=False, **kwargs):
        """
        Evaluate a retained surrogate model.

        The adaptive surrogate may retain only a subset of the surrogate objects
        generated during training. This method evaluates one of the retained
        surrogates selected by ``surrogate_index``.

        Supported ``surrogate_index`` values are:

        * ``-1``: evaluate the last retained surrogate.
        * ``"best"``: evaluate the best retained surrogate according to the
          active storage metric. This is the default.
        * ``"latest"``: evaluate the most recent retained surrogate.
        * a retained adaptive iteration index.
        * a positional integer index into the retained surrogate collection.

        The accepted parameter calling patterns are the same as the underlying
        surrogate type. In general, users may call the surrogate with positional
        arguments, keyword arguments, a parameter dictionary, or a batch array,
        depending on the surrogate implementation.

        :param args: Positional parameter values or batch parameter array.
        :type args: tuple

        :param surrogate_index: Retained surrogate selector.
        :type surrogate_index: int or str

        :param batch_evaluate: If ``True``, interpret the input as a batch of
            parameter values.
        :type batch_evaluate: bool

        :param transpose: If ``True``, transpose array input before forwarding
            it to the retained surrogate.
        :type transpose: bool

        :param kwargs: Keyword parameter values.
        :type kwargs: dict

        :return: Surrogate prediction dictionary.
        :rtype: dict

        :raises RuntimeError: If no surrogate objects have been retained.
        :raises RuntimeError: If the supplied parameter values do not match the
            surrogate calling convention.
        """
        surrogate = self._select_surrogate(surrogate_index)
        return self._evaluate_surrogate_object(
            surrogate, *args, batch_evaluate=batch_evaluate,
            transpose=transpose, **kwargs
        )

    def _evaluate_test_predictions(self, surrogate_index="best"):
        surrogate = self._select_surrogate(surrogate_index)
        return self._evaluate_surrogate_object(
            surrogate,
            self._test_params,
            batch_evaluate=True,
        )

    def _get_target_prediction(self, prediction_data):
        if self._target_field_name not in prediction_data:
            raise RuntimeError(
                f"Surrogate evaluation did not return target field "
                f"'{self._target_field_name}'. Returned fields are "
                f"{list(prediction_data.keys())}."
            )

        return prediction_data[self._target_field_name]

    def _get_test_response_array(self):
        return _as_2d_response_array(self._test_responses)

    def _match_prediction_shape(self, predictions, test_responses):
        predictions = _as_2d_response_array(predictions)

        if predictions.shape == test_responses.shape:
            return predictions

        if predictions.T.shape == test_responses.shape:
            return predictions.T

        raise self._prediction_shape_error(predictions, test_responses)

    def _prediction_shape_error(self, predictions, test_responses):
        return RuntimeError(
            "Surrogate predictions do not match stored test-response shape. "
            f"Prediction shape: {predictions.shape}. "
            f"Test-response shape: {test_responses.shape}."
        )

    def _get_test_prediction_array(self, surrogate_index="best"):
        prediction_data = self._evaluate_test_predictions(surrogate_index)
        predictions = self._get_target_prediction(prediction_data)
        test_responses = self._get_test_response_array()
        return self._match_prediction_shape(predictions, test_responses)

    def _get_plot_sample_indices(self, sample_indices):
        n_samples = self._get_test_response_array().shape[0]
        return _get_sample_indices(sample_indices, n_samples)

    def _get_independent_variable_array(self):
        return np.asarray(self._indep_variable_values)

    def _validate_independent_variable_length(self, responses):
        indep_values = self._get_independent_variable_array()

        if indep_values.size != responses.shape[1]:
            raise RuntimeError(
                "The number of independent-variable values does not match "
                "the number of response values. "
                f"Independent-variable length: {indep_values.size}. "
                f"Response length: {responses.shape[1]}."
            )

    def _get_test_plot_arrays(self, surrogate_index, sample_indices):
        test = self._get_test_response_array()
        prediction = self._get_test_prediction_array(surrogate_index)
        indices = self._get_plot_sample_indices(sample_indices)
        self._validate_independent_variable_length(test)
        return test, prediction, indices

    def _make_response_plot_labels(
        self,
        xlabel,
        ylabel,
        independent_variable_units,
        target_field_units,
    ):
        if xlabel is None:
            xlabel = self._indep_variable_name
        if ylabel is None:
            ylabel = self._target_field_name

        xlabel = _format_axis_label(xlabel, independent_variable_units)
        ylabel = _format_axis_label(ylabel, target_field_units)
        return xlabel, ylabel

    def _make_response_plot_title(self, title):
        if title is not None:
            return title
        target = _replace_underscores(self._target_field_name)
        return f"Surrogate vs. test data: {target}"

    def _plot_one_test_response(self, axes, x_values, y_values, style):
        axes.plot(x_values, y_values, **style)

    def _plot_one_surrogate_response(self, axes, x_values, y_values, style):
        axes.plot(x_values, y_values, **style)

    def _plot_response_sample(
        self,
        axes,
        x_values,
        test_values,
        surrogate_values,
        curve_index,
        test_style,
        surrogate_style,
    ):
        test_style = _add_label_to_first_curve(test_style, "test data", curve_index)
        surrogate_style = _add_label_to_first_curve(surrogate_style, "surrogate", curve_index)
        self._plot_one_test_response(axes, x_values, test_values, test_style)
        self._plot_one_surrogate_response(axes, x_values, surrogate_values, surrogate_style)

    def _plot_response_samples(
        self,
        axes,
        test_responses,
        surrogate_predictions,
        sample_indices,
        test_style,
        surrogate_style,
    ):
        x_values = self._get_independent_variable_array()

        for curve_index, sample_index in enumerate(sample_indices):
            self._plot_response_sample(
                axes,
                x_values,
                test_responses[sample_index, :],
                surrogate_predictions[sample_index, :],
                curve_index,
                test_style.copy(),
                surrogate_style.copy(),
            )

    def _raw_surrogate_error(self, surrogate_index):
        test = self._get_test_response_array()
        prediction = self._get_test_prediction_array(surrogate_index)
        return prediction - test

    def _transform_error(self, raw_error, error_type):
        if error_type == "absolute":
            return np.abs(raw_error)

        if error_type == "squared":
            return raw_error ** 2

        return raw_error

    def _get_surrogate_error_array(self, surrogate_index, error_type):
        raw_error = self._raw_surrogate_error(surrogate_index)
        return self._transform_error(raw_error, error_type)

    def _reduce_error_array(self, errors, error_statistic):
        if error_statistic == "mean":
            return np.nanmean(errors, axis=0)

        if error_statistic == "median":
            return np.nanmedian(errors, axis=0)

        if error_statistic == "max":
            return np.nanmax(errors, axis=0)

        return errors

    def _make_error_label(self, error_type, error_statistic):
        if error_statistic is None:
            return "surrogate error"

        if error_statistic == "max":
            statistic = "maximum"
        else:
            statistic = error_statistic

        return f"{statistic} {error_type} error"

    def _make_error_ylabel(self, error_type):
        target = self._target_field_name

        if error_type == "absolute":
            return f"{target}_absolute_error"

        if error_type == "squared":
            return f"{target}_squared_error"

        return f"{target}_error"

    def _make_error_plot_title(self, title):
        if title is not None:
            return title

        indep_var = _replace_underscores(self._indep_variable_name)
        return f"Surrogate error vs. {indep_var}"

    def _plot_reduced_error(
        self,
        axes,
        x_values,
        selected_errors,
        error_statistic,
        error_type,
        error_style,
    ):
        reduced_errors = self._reduce_error_array(selected_errors, error_statistic)
        error_style.setdefault("label", self._make_error_label(error_type, error_statistic))
        axes.plot(x_values, reduced_errors, **error_style)

    def _plot_individual_error_curves(self, axes, x_values, errors, sample_indices, error_style):
        for curve_index, sample_index in enumerate(sample_indices):
            style = _add_label_to_first_curve(error_style, "surrogate error", curve_index)
            axes.plot(x_values, errors[sample_index, :], **style)

    def _plot_surrogate_errors(
        self,
        axes,
        errors,
        sample_indices,
        error_type,
        error_statistic,
        error_style,
    ):
        x_values = self._get_independent_variable_array()

        if error_statistic is None:
            self._plot_individual_error_curves(axes, x_values, errors, sample_indices, error_style)
            return

        selected_errors = errors[sample_indices, :]
        self._plot_reduced_error(
            axes,
            x_values,
            selected_errors,
            error_statistic,
            error_type,
            error_style,
        )

    def _history_by_metric(self):
        return {
            "rmse": self._root_mean_squared_errors,
            "max_error": self._max_errors,
            "r2": self._r2_scores,
            "score": self._r2_scores,
        }

    def _normalize_history_metrics(self, metrics):
        if isinstance(metrics, str):
            return (metrics,)
        return tuple(metrics)

    def _validate_history_metric(self, metric):
        check_value_is_nonempty_str(metric, "metric")
        metric = metric.lower().strip()

        if metric not in self._history_by_metric():
            raise ValueError(
                "Unsupported error-history metric. Supported metrics are "
                "'rmse', 'max_error', 'r2', and 'score'. "
                f"Received '{metric}'."
            )

        return metric

    def _get_metric_history(self, metric):
        metric = self._validate_history_metric(metric)
        return metric, np.asarray(self._history_by_metric()[metric], dtype=float)

    def _validate_history_length(self, metric, values):
        n_samples = len(self._sample_counts)

        if len(values) != n_samples:
            raise RuntimeError(
                f"History for metric '{metric}' has length {len(values)}, "
                f"but sample-count history has length {n_samples}."
            )

    def _get_metric_plot_style(self, metric, metric_styles):
        default_styles = _default_error_history_styles()
        style = default_styles[metric].copy()
        style.update(metric_styles.get(metric, {}))
        return style

    def _plot_one_metric_history(self, axes, metric, metric_styles):
        metric, values = self._get_metric_history(metric)
        self._validate_history_length(metric, values)
        style = self._get_metric_plot_style(metric, metric_styles)
        axes.plot(self._sample_counts, values, **style)

    def _plot_metric_histories(self, axes, metrics, metric_styles):
        for metric in self._normalize_history_metrics(metrics):
            self._plot_one_metric_history(axes, metric, metric_styles)

    def plot_error_history(
        self,
        metrics=("rmse", "max_error"),
        figure=None,
        axes=None,
        metric_styles=None,
        xlabel=None,
        ylabel=None,
        title=None,
        xlim=None,
        ylim=None,
        xscale=None,
        yscale=None,
        sample_count_units=None,
        error_units=None,
        grid=True,
        show_legend=True,
    ):
        """
        Plot adaptive-surrogate error histories versus number of training
        samples.

        This method plots stored adaptive-surrogate score histories against the
        number of training samples used at each adaptive-training batch. The
        primary intended use is to visualize convergence of the adaptive
        surrogate as additional training samples are added.

        Supported metrics are:

        * ``"rmse"``: root-mean-squared error history
        * ``"max_error"``: maximum absolute error history
        * ``"r2"``: :math:`R^2` score history
        * ``"score"``: alias for ``"r2"``

        Axis labels automatically replace underscores with spaces. For example,
        ``"number_of_training_samples"`` is displayed as
        ``"number of training samples"``. If units are supplied, they are
        appended in parentheses.

        :param metrics: Metric or metrics to plot. A single metric may be passed
            as a string. Multiple metrics may be passed as a sequence of strings.
            Defaults to ``("rmse", "max_error")``.
        :type metrics: str or sequence[str]

        :param figure: Optional Matplotlib figure. If provided and ``axes`` is
            not provided, the first axes in the figure is used. If the figure has
            no axes, a new axes is added.
        :type figure: matplotlib.figure.Figure or None

        :param axes: Optional Matplotlib axes to draw on. If provided, this takes
            precedence over ``figure``.
        :type axes: matplotlib.axes.Axes or None

        :param metric_styles: Optional mapping from metric name to keyword
            arguments passed to :meth:`matplotlib.axes.Axes.plot`. User-supplied
            style values override the default style for each metric.

            Example::

                {
                    "rmse": {
                        "color": "tab:blue",
                        "linestyle": "-",
                        "marker": "o",
                    },
                    "max_error": {
                        "color": "tab:red",
                        "linestyle": "--",
                    },
                }

        :type metric_styles: dict or None

        :param xlabel: Optional x-axis label. If ``None``,
            ``"number_of_training_samples"`` is used.
        :type xlabel: str or None

        :param ylabel: Optional y-axis label. If ``None``,
            ``"error_or_score"`` is used.
        :type ylabel: str or None

        :param title: Optional plot title. If ``None``, a default title is used.
        :type title: str or None

        :param xlim: Optional x-axis limits, e.g. ``(0, 100)``.
        :type xlim: tuple or list or None

        :param ylim: Optional y-axis limits, e.g. ``(1e-4, 1)``.
        :type ylim: tuple or list or None

        :param xscale: Optional x-axis scale, e.g. ``"linear"`` or ``"log"``.
        :type xscale: str or None

        :param yscale: Optional y-axis scale, e.g. ``"linear"`` or ``"log"``.
        :type yscale: str or None

        :param sample_count_units: Optional units appended to the x-axis label.
            This is usually ``None`` because sample counts are dimensionless.
        :type sample_count_units: str or None

        :param error_units: Optional units appended to the y-axis label.
        :type error_units: str or None

        :param grid: If not ``None``, passed to
            :meth:`matplotlib.axes.Axes.grid`.
        :type grid: bool or None

        :param show_legend: If ``True``, show the axes legend.
        :type show_legend: bool

        :return: Matplotlib ``(figure, axes)`` pair.
        :rtype: tuple

        :raises ValueError: If an unsupported metric is requested.
        :raises RuntimeError: If a metric history length does not match the
            sample-count history length.

        **Example**

        .. code-block:: python

            fig, ax = study.surrogate.plot_error_history(
                metrics=("rmse", "max_error"),
                error_units="K",
                yscale="log",
                metric_styles={
                    "rmse": {"color": "tab:blue", "marker": "o"},
                    "max_error": {"color": "tab:red", "linestyle": "--"},
                },
            )
        """
        figure, axes = _get_or_create_matplotlib_axes(figure, axes)

        if metric_styles is None:
            metric_styles = {}

        self._plot_metric_histories(axes, metrics, metric_styles)

        if xlabel is None:
            xlabel = "number_of_training_samples"
        if ylabel is None:
            ylabel = "error_or_score"
        if title is None:
            title = "Adaptive surrogate error history"

        xlabel = _format_axis_label(xlabel, sample_count_units)
        ylabel = _format_axis_label(ylabel, error_units)

        _apply_axes_plot_options(
            axes,
            xlabel,
            ylabel,
            title,
            xlim,
            ylim,
            xscale,
            yscale,
            grid,
        )
        _apply_legend(axes, show_legend)

        return figure, axes

    def plot_surrogate_error_vs_independent_variable(
        self,
        surrogate_index="best",
        sample_indices=None,
        error_type="absolute",
        error_statistic="mean",
        figure=None,
        axes=None,
        error_style=None,
        xlabel=None,
        ylabel=None,
        title=None,
        xlim=None,
        ylim=None,
        xscale=None,
        yscale=None,
        independent_variable_units=None,
        target_field_units=None,
        error_units=None,
        grid=True,
        show_legend=True,
    ):
        """
        Plot surrogate prediction error versus the surrogate independent
        variable.

        This method evaluates a retained surrogate at the stored test-parameter
        locations, compares the surrogate predictions to the stored test
        responses, and plots the error as a function of the independent
        variable.

        The raw error is defined as

        .. math::

            e = \\hat{y} - y

        where (\\hat{y}) is the surrogate prediction and (y) is the stored
        test response.

        Supported error definitions are:

        * ``"absolute"``: plots ``abs(surrogate - test)``
        * ``"signed"``: plots ``surrogate - test``
        * ``"squared"``: plots ``(surrogate - test)**2``

        By default, the method plots the mean absolute error over the selected
        test samples at each independent-variable location. Set
        ``error_statistic=None`` to plot one error curve per selected test
        sample.

        Axis labels automatically replace underscores with spaces. For example,
        ``"temperature_absolute_error"`` is displayed as
        ``"temperature absolute error"``. If units are supplied, they are
        appended in parentheses.

        :param surrogate_index: Retained surrogate selector. Supported values
            are ``"best"``, ``"latest"``, a retained adaptive iteration index,
            or a positional retained-surrogate index. Defaults to ``"best"``.
        :type surrogate_index: int or str

        :param sample_indices: Optional subset of test-sample indices to use.
            If ``None``, all stored test samples are used.
        :type sample_indices: array-like of int or None

        :param error_type: Error definition. Must be one of ``"absolute"``,
            ``"signed"``, or ``"squared"``.
        :type error_type: str

        :param error_statistic: Statistic used to reduce the selected test
            samples at each independent-variable location. Must be one of
            ``"mean"``, ``"median"``, ``"max"``, or ``None``. If ``None``, one
            error curve is plotted per selected test sample.
        :type error_statistic: str or None

        :param figure: Optional Matplotlib figure. If provided and ``axes`` is
            not provided, the first axes in the figure is used. If the figure has
            no axes, a new axes is added.
        :type figure: matplotlib.figure.Figure or None

        :param axes: Optional Matplotlib axes to draw on. If provided, this takes
            precedence over ``figure``.
        :type axes: matplotlib.axes.Axes or None

        :param error_style: Optional keyword arguments passed to
            :meth:`matplotlib.axes.Axes.plot` for the plotted error curve or
            curves. These values override the default error style.
        :type error_style: dict or None

        :param xlabel: Optional x-axis label. If ``None``, the adaptive
            surrogate's independent-variable name is used.
        :type xlabel: str or None

        :param ylabel: Optional y-axis label. If ``None``, a label is generated
            from the target-field name and ``error_type``.
        :type ylabel: str or None

        :param title: Optional plot title. If ``None``, a default title is used.
        :type title: str or None

        :param xlim: Optional x-axis limits, e.g. ``(0, 1)``.
        :type xlim: tuple or list or None

        :param ylim: Optional y-axis limits, e.g. ``(-1, 1)``.
        :type ylim: tuple or list or None

        :param xscale: Optional x-axis scale, e.g. ``"linear"`` or ``"log"``.
        :type xscale: str or None

        :param yscale: Optional y-axis scale, e.g. ``"linear"`` or ``"log"``.
        :type yscale: str or None

        :param independent_variable_units: Optional units appended to the x-axis
            label.
        :type independent_variable_units: str or None

        :param target_field_units: Optional units for the target field. If
            ``error_units`` is not provided, these units are used for
            ``"absolute"`` and ``"signed"`` errors. For ``"squared"`` errors,
            ``"^2"`` is appended to these units.
        :type target_field_units: str or None

        :param error_units: Optional units appended to the y-axis label. If
            provided, this overrides units inferred from ``target_field_units``.
        :type error_units: str or None

        :param grid: If not ``None``, passed to
            :meth:`matplotlib.axes.Axes.grid`.
        :type grid: bool or None

        :param show_legend: If ``True``, show the axes legend.
        :type show_legend: bool

        :return: Matplotlib ``(figure, axes)`` pair.
        :rtype: tuple

        :raises RuntimeError: If no retained surrogate is available, if the
            retained surrogate does not return the target field, or if the
            prediction shape is incompatible with the stored test responses.
        :raises ValueError: If ``error_type`` or ``error_statistic`` is invalid,
            or if ``sample_indices`` contains an invalid test-sample index.

        **Examples**

        Plot mean absolute error:

        .. code-block:: python

            fig, ax = study.surrogate.plot_surrogate_error_vs_independent_variable(
                error_type="absolute",
                error_statistic="mean",
                independent_variable_units="s",
                target_field_units="K",
            )

        Plot individual signed error curves:

        .. code-block:: python

            fig, ax = study.surrogate.plot_surrogate_error_vs_independent_variable(
                error_type="signed",
                error_statistic=None,
                sample_indices=[0, 1, 2],
                target_field_units="MPa",
            )
        """
        error_type = _validate_error_type(error_type)
        error_statistic = _validate_error_statistic(error_statistic)
        figure, axes = _get_or_create_matplotlib_axes(figure, axes)

        errors = self._get_surrogate_error_array(surrogate_index, error_type)
        self._validate_independent_variable_length(errors)
        indices = self._get_plot_sample_indices(sample_indices)

        error_style = _merge_plot_style(_default_surrogate_error_style(), error_style)
        self._plot_surrogate_errors(
            axes,
            errors,
            indices,
            error_type,
            error_statistic,
            error_style,
        )

        if xlabel is None:
            xlabel = self._indep_variable_name
        if ylabel is None:
            ylabel = self._make_error_ylabel(error_type)

        error_units = _error_units_for_type(error_type, target_field_units, error_units)
        xlabel = _format_axis_label(xlabel, independent_variable_units)
        ylabel = _format_axis_label(ylabel, error_units)
        title = self._make_error_plot_title(title)

        _apply_axes_plot_options(
            axes,
            xlabel,
            ylabel,
            title,
            xlim,
            ylim,
            xscale,
            yscale,
            grid,
        )
        _apply_legend(axes, show_legend)

        return figure, axes

    def plot_surrogate_vs_test_data(
        self,
        surrogate_index="best",
        sample_indices=None,
        figure=None,
        axes=None,
        test_style=None,
        surrogate_style=None,
        xlabel=None,
        ylabel=None,
        title=None,
        xlim=None,
        ylim=None,
        xscale=None,
        yscale=None,
        independent_variable_units=None,
        target_field_units=None,
        grid=True,
        show_legend=True,
    ):
        """
        Plot retained surrogate predictions and stored test data versus the
        surrogate independent variable.

        This method evaluates a retained surrogate at the adaptive surrogate's
        stored test-parameter locations and plots the resulting surrogate
        response curves alongside the corresponding test-data response curves.
        One test-data curve and one surrogate curve are plotted for each
        selected test sample.

        Matplotlib is imported lazily when this method is called, so importing
        :mod:`matcal.core.adaptive_surrogates` does not require Matplotlib unless
        plotting is requested.

        Axis labels automatically replace underscores with spaces. For example,
        ``"target_field"`` is displayed as ``"target field"``. If units are
        supplied, they are appended in parentheses.

        :param surrogate_index: Retained surrogate selector. Supported values
            are ``"best"``, ``"latest"``, a retained adaptive iteration index,
            or a positional retained-surrogate index. Defaults to ``"best"``.
        :type surrogate_index: int or str

        :param sample_indices: Optional subset of test-sample indices to plot.
            If ``None``, all stored test samples are plotted.
        :type sample_indices: array-like of int or None

        :param figure: Optional Matplotlib figure. If provided and ``axes`` is
            not provided, the first axes in the figure is used. If the figure has
            no axes, a new axes is added.
        :type figure: matplotlib.figure.Figure or None

        :param axes: Optional Matplotlib axes to draw on. If provided, this takes
            precedence over ``figure``.
        :type axes: matplotlib.axes.Axes or None

        :param test_style: Optional keyword arguments passed to
            :meth:`matplotlib.axes.Axes.plot` for the test-data curves. These
            values override the default test-data style.
        :type test_style: dict or None

        :param surrogate_style: Optional keyword arguments passed to
            :meth:`matplotlib.axes.Axes.plot` for the surrogate-prediction
            curves. These values override the default surrogate style.
        :type surrogate_style: dict or None

        :param xlabel: Optional x-axis label. If ``None``, the adaptive
            surrogate's independent-variable name is used.
        :type xlabel: str or None

        :param ylabel: Optional y-axis label. If ``None``, the adaptive
            surrogate's target-field name is used.
        :type ylabel: str or None

        :param title: Optional plot title. If ``None``, a default title is used.
        :type title: str or None

        :param xlim: Optional x-axis limits, e.g. ``(0, 1)``.
        :type xlim: tuple or list or None

        :param ylim: Optional y-axis limits, e.g. ``(-1, 1)``.
        :type ylim: tuple or list or None

        :param xscale: Optional x-axis scale, e.g. ``"linear"`` or ``"log"``.
        :type xscale: str or None

        :param yscale: Optional y-axis scale, e.g. ``"linear"`` or ``"log"``.
        :type yscale: str or None

        :param independent_variable_units: Optional units appended to the x-axis
            label.
        :type independent_variable_units: str or None

        :param target_field_units: Optional units appended to the y-axis label.
        :type target_field_units: str or None

        :param grid: If not ``None``, passed to
            :meth:`matplotlib.axes.Axes.grid`.
        :type grid: bool or None

        :param show_legend: If ``True``, show the axes legend.
        :type show_legend: bool

        :return: Matplotlib ``(figure, axes)`` pair.
        :rtype: tuple

        :raises RuntimeError: If no retained surrogate is available, if the
            retained surrogate does not return the target field, or if the
            prediction shape is incompatible with the stored test responses.
        :raises ValueError: If ``sample_indices`` contains an invalid test-sample
            index.

        **Example**

        .. code-block:: python

            fig, ax = study.surrogate.plot_surrogate_vs_test_data(
                sample_indices=[0, 1, 2],
                independent_variable_units="s",
                target_field_units="K",
                test_style={"color": "black", "marker": "o"},
                surrogate_style={"color": "tab:red", "linestyle": "--"},
            )
        """
        figure, axes = _get_or_create_matplotlib_axes(figure, axes)

        test, prediction, indices = self._get_test_plot_arrays(
            surrogate_index,
            sample_indices,
        )

        test_style = _merge_plot_style(_default_test_data_style(), test_style)
        surrogate_style = _merge_plot_style(_default_surrogate_style(), surrogate_style)

        self._plot_response_samples(
            axes,
            test,
            prediction,
            indices,
            test_style,
            surrogate_style,
        )

        xlabel, ylabel = self._make_response_plot_labels(
            xlabel,
            ylabel,
            independent_variable_units,
            target_field_units,
        )
        title = self._make_response_plot_title(title)

        _apply_axes_plot_options(
            axes,
            xlabel,
            ylabel,
            title,
            xlim,
            ylim,
            xscale,
            yscale,
            grid,
        )
        _apply_legend(axes, show_legend)

        return figure, axes

    def plot_worst_N(
        self,
        N=5,
        n_figures=1,
        surrogate_index="best",
        metric="max_error",
        error_type="signed",
        test_style=None,
        surrogate_style=None,
        error_style=None,
        independent_variable_units=None,
        target_field_units=None,
        error_units=None,
        grid=True,
        show_legend=True,
    ):
        """
        Plot surrogate predictions and errors for the worst N stored test samples.

        The worst samples are identified using :meth:`test_sample_errors`.

        The selected samples are split across ``n_figures`` figures. Each figure
        contains two axes:

        * the left axis compares stored test data and retained-surrogate
          predictions for that figure's subset of samples;
        * the right axis plots the corresponding surrogate errors.

        Axis limits are kept common across all generated figures so that the
        response and error plots can be compared directly.

        :param N: Number of worst test samples to plot.
        :type N: int

        :param n_figures: Number of figures to split the selected samples across.
            If ``n_figures`` is larger than the number of plotted samples, only
            one figure per sample is created.
        :type n_figures: int

        :param surrogate_index: Retained surrogate selector. Defaults to
            ``"best"``.
        :type surrogate_index: int or str

        :param metric: Per-sample error metric used to rank the worst samples.
            Supported values are ``"max_error"``, ``"max_abs_error"``,
            ``"linf"``, ``"rmse"``, ``"mae"``, and ``"mean_abs_error"``.
        :type metric: str

        :param error_type: Error definition for the right-axis plots. Must be
            ``"signed"``, ``"absolute"``, or ``"squared"``.
        :type error_type: str

        :param test_style: Optional style for test-data curves.
        :type test_style: dict or None

        :param surrogate_style: Optional style for surrogate-prediction curves.
        :type surrogate_style: dict or None

        :param error_style: Optional style for error curves.
        :type error_style: dict or None

        :param independent_variable_units: Optional x-axis units.
        :type independent_variable_units: str or None

        :param target_field_units: Optional target-field units.
        :type target_field_units: str or None

        :param error_units: Optional error units. If omitted, inferred from
            ``target_field_units``.
        :type error_units: str or None

        :param grid: If not ``None``, passed to
            :meth:`matplotlib.axes.Axes.grid`.
        :type grid: bool or None

        :param show_legend: If ``True``, show legends on each figure.
        :type show_legend: bool

        :return: ``(figures, axes_groups, worst_indices)``, where ``figures`` is
            a list of Matplotlib figures, ``axes_groups`` is a list of
            ``(1, 2)`` axes arrays, and ``worst_indices`` contains the plotted
            test-sample indices sorted from worst to best.
        :rtype: tuple
        """
        metric, error_type = self._validate_plot_worst_N_inputs(
            N, n_figures, metric, error_type
        )
        arrays = self._get_plot_worst_N_arrays(surrogate_index, error_type)
        worst_indices, index_groups, sample_errors = self._get_plot_worst_N_groups(
            N, n_figures, surrogate_index, metric
        )
        labels = self._get_plot_worst_N_labels(
            error_type, independent_variable_units, target_field_units, error_units
        )
        styles = self._get_plot_worst_N_styles(
            test_style, surrogate_style, error_style
        )
        limits = self._get_plot_worst_N_limits(arrays, worst_indices, error_type)
        figures, axes_groups = self._make_plot_worst_N_figures(
            index_groups, arrays, labels, styles, limits, sample_errors,
            metric, error_type, surrogate_index, grid, show_legend
        )
        return figures, axes_groups, worst_indices

    def plot_worst_n(self, n=5, **kwargs):
        """
        Lowercase alias for :meth:`plot_worst_N`.
        """
        return self.plot_worst_N(N=n, **kwargs)

    def _validate_plot_worst_N_inputs(self, N, n_figures, metric, error_type):
        check_value_is_positive_integer(N, "N")
        check_value_is_positive_integer(n_figures, "n_figures")
        metric = self._validate_test_sample_error_metric(metric)
        error_type = _validate_error_type(error_type)
        return metric, error_type

    def _get_plot_worst_N_arrays(self, surrogate_index, error_type):
        arrays = {}
        arrays["test"] = self._get_test_response_array()
        arrays["prediction"] = self._get_test_prediction_array(surrogate_index)
        arrays["error"] = self._get_surrogate_error_array(surrogate_index, error_type)
        arrays["x"] = self._get_independent_variable_array()
        self._validate_plot_worst_N_array_lengths(arrays)
        return arrays

    def _validate_plot_worst_N_array_lengths(self, arrays):
        self._validate_independent_variable_length(arrays["test"])
        self._validate_independent_variable_length(arrays["prediction"])
        self._validate_independent_variable_length(arrays["error"])

    def _get_plot_worst_N_groups(self, N, n_figures, surrogate_index, metric):
        worst_indices = self.worst_test_sample_indices(
            n=N, surrogate_index=surrogate_index, metric=metric
        )
        sample_errors = self.test_sample_errors(
            surrogate_index=surrogate_index, metric=metric
        )
        n_figures_to_make = min(n_figures, len(worst_indices))
        index_groups = np.array_split(worst_indices, n_figures_to_make)
        return worst_indices, index_groups, sample_errors

    def _get_plot_worst_N_labels(
        self, error_type, independent_variable_units,
        target_field_units, error_units
    ):
        labels = {}
        labels["x"] = _format_axis_label(
            self._indep_variable_name, independent_variable_units
        )
        labels["response_y"] = _format_axis_label(
            self._target_field_name, target_field_units
        )
        error_units = _error_units_for_type(
            error_type, target_field_units, error_units
        )
        labels["error_y"] = _format_axis_label(
            self._make_error_ylabel(error_type), error_units
        )
        return labels

    def _get_plot_worst_N_styles(self, test_style, surrogate_style, error_style):
        styles = {}
        styles["test"] = _merge_plot_style(
            self._default_plot_worst_N_test_style(), test_style
        )
        styles["surrogate"] = _merge_plot_style(
            self._default_plot_worst_N_surrogate_style(), surrogate_style
        )
        styles["error"] = _merge_plot_style(
            self._default_plot_worst_N_error_style(), error_style
        )
        return styles

    def _default_plot_worst_N_test_style(self):
        return {
            "linestyle": "-",
            "marker": None,
            "alpha": 0.75,
            "linewidth": 1.5,
        }

    def _default_plot_worst_N_surrogate_style(self):
        return {
            "linestyle": "--",
            "marker": None,
            "alpha": 0.9,
            "linewidth": 1.8,
        }

    def _default_plot_worst_N_error_style(self):
        return {
            "linestyle": "-",
            "marker": None,
            "alpha": 0.9,
            "linewidth": 1.8,
        }

    def _get_plot_worst_N_limits(self, arrays, worst_indices, error_type):
        limits = {}
        limits["x"] = self._get_padded_plot_limits(arrays["x"])
        limits["response_y"] = self._get_plot_worst_N_response_limits(
            arrays, worst_indices
        )
        limits["error_y"] = self._get_padded_plot_limits(
            arrays["error"][worst_indices, :].ravel(),
            include_zero=error_type == "signed",
        )
        return limits

    def _get_plot_worst_N_response_limits(self, arrays, worst_indices):
        values = np.concatenate((
            arrays["test"][worst_indices, :].ravel(),
            arrays["prediction"][worst_indices, :].ravel(),
        ))
        return self._get_padded_plot_limits(values)

    def _get_padded_plot_limits(self, values, include_zero=False):
        values = np.asarray(values, dtype=float).ravel()
        values = values[np.isfinite(values)]
        if include_zero:
            values = np.concatenate((values, np.array([0.0])))
        return self._padded_limits_from_finite_values(values)

    def _padded_limits_from_finite_values(self, values):
        if values.size == 0:
            return None
        lower = np.nanmin(values)
        upper = np.nanmax(values)
        pad = self._plot_limit_padding(lower, upper)
        return lower - pad, upper + pad

    def _plot_limit_padding(self, lower, upper):
        if np.isclose(lower, upper):
            return 0.05 * max(abs(lower), 1.0)
        return 0.05 * (upper - lower)

    def _make_plot_worst_N_figures(
        self, index_groups, arrays, labels, styles, limits, sample_errors,
        metric, error_type, surrogate_index, grid, show_legend
    ):
        figures = []
        axes_groups = []
        rank_start = 1
        for group_indices in index_groups:
            figure, axes = self._make_one_plot_worst_N_figure(
                group_indices, rank_start, arrays, labels, styles, limits,
                sample_errors, metric, error_type, surrogate_index, grid, show_legend
            )
            figures.append(figure)
            axes_groups.append(axes)
            rank_start += len(group_indices)
        return figures, axes_groups

    def _make_one_plot_worst_N_figure(
        self, group_indices, rank_start, arrays, labels, styles, limits,
        sample_errors, metric, error_type, surrogate_index, grid, show_legend
    ):
        figure, axes = self._create_plot_worst_N_figure_and_axes()
        self._plot_worst_N_group_curves(
            axes, group_indices, rank_start, arrays, styles,
            sample_errors, metric, error_type
        )
        self._decorate_plot_worst_N_figure(
            figure, axes, rank_start, group_indices, labels, limits,
            error_type, surrogate_index, grid, show_legend
        )
        return figure, axes

    def _create_plot_worst_N_figure_and_axes(self):
        import matplotlib.pyplot as plt

        return plt.subplots(
            1, 2, squeeze=False, figsize=(13, 5), constrained_layout=True
        )

    def _plot_worst_N_group_curves(
        self, axes, group_indices, rank_start, arrays, styles,
        sample_errors, metric, error_type
    ):
        for local_index, sample_index in enumerate(group_indices):
            self._plot_one_worst_N_sample(
                axes, local_index, sample_index, rank_start, arrays,
                styles, sample_errors, metric
            )
        if error_type == "signed":
            self._add_plot_worst_N_zero_error_line(axes[0, 1])

    def _plot_one_worst_N_sample(
        self, axes, local_index, sample_index, rank_start, arrays,
        styles, sample_errors, metric
    ):
        rank = rank_start + local_index
        color = f"C{local_index % 10}"
        self._plot_one_worst_N_response(
            axes[0, 0], sample_index, rank, color,
            arrays, styles, sample_errors, metric
        )
        self._plot_one_worst_N_error(
            axes[0, 1], sample_index, color, arrays, styles
        )

    def _plot_one_worst_N_response(
        self, axes, sample_index, rank, color,
        arrays, styles, sample_errors, metric
    ):
        test_style, surrogate_style = self._make_worst_N_response_styles(
            sample_index, rank, color, styles, sample_errors, metric
        )
        axes.plot(arrays["x"], arrays["test"][sample_index, :], **test_style)
        axes.plot(
            arrays["x"], arrays["prediction"][sample_index, :], **surrogate_style
        )

    def _plot_one_worst_N_error(
        self, axes, sample_index, color, arrays, styles
    ):
        error_style = self._make_worst_N_error_style(
            sample_index, color, styles
        )
        axes.plot(arrays["x"], arrays["error"][sample_index, :], **error_style)

    def _make_worst_N_response_styles(
        self, sample_index, rank, color, styles, sample_errors, metric
    ):
        test_style = self._make_worst_N_test_style(
            sample_index, rank, color, styles, sample_errors, metric
        )
        surrogate_style = styles["surrogate"].copy()
        surrogate_style.setdefault("color", color)
        surrogate_style["label"] = f"surrogate sample {sample_index}"
        return test_style, surrogate_style

    def _make_worst_N_test_style(
        self, sample_index, rank, color, styles, sample_errors, metric
    ):
        style = styles["test"].copy()
        style.setdefault("color", color)
        style["label"] = self._make_worst_N_test_label(
            sample_index, rank, sample_errors[sample_index], metric
        )
        return style

    def _make_worst_N_test_label(self, sample_index, rank, sample_error, metric):
        return (
            f"test sample {sample_index} "
            f"(rank {rank}, {metric}={sample_error:.4g})"
        )

    def _make_worst_N_error_style(self, sample_index, color, styles):
        style = styles["error"].copy()
        style.setdefault("color", color)
        style["label"] = f"sample {sample_index}"
        return style

    def _add_plot_worst_N_zero_error_line(self, axes):
        axes.axhline(
            0.0, color="black", linewidth=0.8,
            alpha=0.7, label="_nolegend_"
        )

    def _decorate_plot_worst_N_figure(
        self, figure, axes, rank_start, group_indices, labels, limits,
        error_type, surrogate_index, grid, show_legend
    ):
        rank_end = rank_start + len(group_indices) - 1
        self._decorate_plot_worst_N_response_axes(
            axes[0, 0], rank_start, rank_end, labels, limits, grid
        )
        self._decorate_plot_worst_N_error_axes(
            axes[0, 1], rank_start, rank_end, labels, limits, error_type, grid
        )
        self._decorate_plot_worst_N_figure_title(figure, surrogate_index)
        self._maybe_add_plot_worst_N_legends(axes, show_legend)

    def _decorate_plot_worst_N_response_axes(
        self, axes, rank_start, rank_end, labels, limits, grid
    ):
        axes.set_xlabel(labels["x"])
        axes.set_ylabel(labels["response_y"])
        axes.set_title(
            f"Worst samples {rank_start}-{rank_end}: surrogate vs. test data"
        )
        self._apply_plot_worst_N_axis_limits(axes, limits["x"], limits["response_y"])
        self._maybe_add_plot_worst_N_grid(axes, grid)

    def _decorate_plot_worst_N_error_axes(
        self, axes, rank_start, rank_end, labels, limits, error_type, grid
    ):
        axes.set_xlabel(labels["x"])
        axes.set_ylabel(labels["error_y"])
        axes.set_title(f"Worst samples {rank_start}-{rank_end}: {error_type} error")
        self._apply_plot_worst_N_axis_limits(axes, limits["x"], limits["error_y"])
        self._maybe_add_plot_worst_N_grid(axes, grid)

    def _apply_plot_worst_N_axis_limits(self, axes, xlim, ylim):
        if xlim is not None:
            axes.set_xlim(xlim)
        if ylim is not None:
            axes.set_ylim(ylim)

    def _maybe_add_plot_worst_N_grid(self, axes, grid):
        if grid is not None:
            axes.grid(grid)

    def _decorate_plot_worst_N_figure_title(self, figure, surrogate_index):
        figure.suptitle(
            f"Worst stored test samples for retained surrogate "
            f"'{surrogate_index}'"
        )

    def _maybe_add_plot_worst_N_legends(self, axes, show_legend):
        if show_legend:
            axes[0, 0].legend()
            axes[0, 1].legend()




class SparseGridAdaptiveSurrogate(AdaptiveSurrogate):
    """
    Adaptive surrogate wrapper for PyApprox sparse-grid surrogate objects.

    The retained surrogate objects are PyApprox fitter results. This subclass
    handles the additional steps required to evaluate those objects:

    * process MatCal-style positional, keyword, dictionary, or batch inputs;
    * check physical parameter bounds;
    * evaluate the PyApprox sparse-grid surrogate in native parameter space; and
    * package the response with the independent-variable values.

    Surrogate-object retention, score histories, test data storage, and
    retained-surrogate metadata are managed by the base
    :class:`AdaptiveSurrogate` class.
    """
    def _evaluate_surrogate_object(self, result, *args, batch_evaluate=False,
                                transpose=True, **kwargs):
        # Stored object is a PyApprox fitter.result().
        surrogate_fun = result.surrogate

        params_array = _process_surrogate_args_call(
            self._param_names, *args,
            batch_evaluate=batch_evaluate, transpose=transpose, **kwargs
        )

        # PyApprox wants shape (nvars, nsamples).
        if not batch_evaluate:
            params_array = np.atleast_2d(params_array).T
        else:
            if params_array.shape[0] != len(self._param_names):
                params_array = params_array.T

        # Range check in physical/native parameter space.
        params_dict = _convert_param_array_to_dict(params_array.T, self._param_names)
        _check_params_in_range(
            params_dict, self._bounds.T,
            self._enforce_training_data_parameter_range
        )

        # PyApprox sparse grids are constructed directly on the physical/native
        # parameter domain. No parameter transform is performed.

        # PyApprox returns (n_qois, nsamples).
        response = surrogate_fun(params_array)
        response = np.asarray(response)

        if response.ndim == 2:
            response = response.T

        if not batch_evaluate:
            response = response.flatten()

        results = {self._target_field_name: response}
        results[self._indep_variable_name] = self._indep_variable_values
        return results

    def __call__(self, *args, surrogate_index="best", batch_evaluate=False,
                 transpose=True, **kwargs):
        """
        Evaluate a retained PyApprox sparse-grid surrogate.

        See :meth:`AdaptiveSurrogate.__call__` for the retained-surrogate
        selection rules. This subclass uses ``transpose=True`` by default
        because PyApprox uses ``(n_parameters, n_samples)`` orientation.

        :param args: Positional parameter values, parameter dictionary, or batch
            parameter array.
        :type args: tuple

        :param surrogate_index: Retained surrogate selector. Accepts ``-1``,
            ``"best"``, ``"latest"``, a retained iteration index, or a
            positional retained-surrogate index. Defaults to ``"best"``.
        :type surrogate_index: int or str

        :param batch_evaluate: If ``True``, evaluate a batch of parameter
            samples.
        :type batch_evaluate: bool

        :param transpose: If ``True``, transpose the input array before
            orientation checks.
        :type transpose: bool

        :param kwargs: Keyword parameter values.
        :type kwargs: dict

        :return: Dictionary containing the target-field prediction and the
            independent-variable values.
        :rtype: dict
        """
        result = self._select_surrogate(surrogate_index)
        return self._evaluate_surrogate_object(
            result, *args, batch_evaluate=batch_evaluate,
            transpose=transpose, **kwargs
        )


class AdaptiveSurrogateStudyBase(HaltonStudy):
    def __init__(self, *parameters):
        super().__init__(*parameters)

        self._bounds = _get_parameter_bounds(self._parameter_collection)

        self._target_field_name = None
        self._independent_variable=None
        self._independent_variable_values=None
        self._evaluation_set_added = False
        self._results_synchronizer = None

        self._surrogate = None
        self._user_test_data = None

        self._max_training_samples=None
        self._training_samples_user_set = False
        self._number_of_test_samples=None
        self._test_samples_user_set = False
        self._training_batch_number = 1
        self.set_max_training_samples()

        self._rmse_goal = None
        self._max_abs_error_goal = None
        self.set_error_stopping_criteria()

        self._surrogate_save_filename = None
        self._test_group_random_seed = None

        self._surrogate_storage_best_n_surrogates = 1
        self._surrogate_storage_every_n_batches = None
        self._surrogate_storage_score_metric = "max_error"

    def set_error_stopping_criteria(self,
                                    rmse_goal: float=1e-2,
                                    max_abs_error_goal: float=1e-1):
        """
        Set the error thresholds that determine when the adaptive surrogate
        training stops.

        When
        the *rmse* falls below ``rmse_goal`` **or** the
        *maximum absolute* error falls below ``max_abs_error_goal`` the training
        loop terminates (provided at least two batches have been evaluated).

        :param rmse_goal: Desired upper bound for the root mean squared
            error. Must be a positive number. 
        :type rmse_goal: float, optional

        :param max_abs_error_goal: Desired upper bound for the maximum absolute
            error. Must be a positive number. 
        :type max_abs_error_goal: float, optional
        """
        check_value_is_positive_real(rmse_goal, "rmse_goal")
        self._rmse_goal = float(rmse_goal)

        check_value_is_positive_real(max_abs_error_goal, "max_abs_error_goal")
        self._max_abs_error_goal = float(max_abs_error_goal)

    def set_independent_variable(self, independent_variable, 
                                 independent_variable_values):
        """
        Specify an independent (auxiliary) variable and the values at which the surrogate
        will be evaluated.

        This variable is **not** a model input; it is a field that will be used later
        (for example, a spatial coordinate, a time step, or any other scalar quantity
        that the surrogate should be conditioned on).  The surrogate will be trained
        on the parameter samples generated by the study and then provide a response at each
        value supplied in ``independent_variable_values``.

        :param independent_variable: Name of the independent variable (e.g. ``"time"``,
            ``"x_position"``, …) that will be attached to the surrogate output.
        :type independent_variable: str
        :param independent_variable_values: A 1‑D array‑like collection of real numbers
            indicating the points at which the surrogate should be queried.
        :type independent_variable_values: array‑like of real numbers
        """
        check_value_is_nonempty_str(independent_variable, "independent_variable")
        self._independent_variable = independent_variable
        check_value_is_array_like_of_reals(independent_variable_values,
                                           "independent_variable_values")
        self._independent_variable_values = independent_variable_values
        logger.debug(f"Independent variable field set to {self._independent_variable}")
        logger.debug(f"Independent variable values set to {self._independent_variable_values}")

    def set_number_of_test_samples(self, number_of_test_samples):
        """
        Set the number of samples that will be used for testing.
        By default we test against ``max_training_samples``/20 or 
        the number of parameters*10, whichever is greater.
        
        :param max_training_samples: desired number of test samples
        :type max_training_samples: int
        """
        self._test_samples_user_set = True
        check_value_is_positive_integer(number_of_test_samples, "number_of_test_samples")
        self._number_of_test_samples = number_of_test_samples
        super().set_number_of_samples(self._number_of_test_samples)
        logger.debug(f"number_of_test_samples set to {self._number_of_test_samples}")

    def set_max_training_samples(self, max_training_samples=1000):
        """
        Set the maximum number of training samples you want to be run for 
        Sparse Grid surrogate generation. If the convergence criteria is not reached, 
        the training for the surrogate will stop after max_training_samples has been 
        reached.
        
        :param max_training_samples: desired maximum number of samples
        :type max_training_samples: int
        """
        check_value_is_positive_integer(max_training_samples, "max_training_samples")
        self._max_training_samples = max_training_samples
        logger.debug(f"max_training_samples set to {self._max_training_samples}")
        if not self._test_samples_user_set:
            self.set_number_of_test_samples(self._set_default_number_of_test_samples())
            self._test_samples_user_set = False

    def set_test_data(self, study_results):
        """
        Provide an external test‑data set for the adaptive surrogate study. This must contain
        the model name and field names necessary for the surrogate. This should only be used 
        when re-running surrogate generation with a previously existing test set from a 
        previous run where surrogate training was attempted. The independent variable data
        must also match what is specified for surrogate training.

        If this method is **not** called, the adaptive study will automatically
        generate a test data set using a Halton sampling design.  The number of
        test samples is taken from the value set via
        :meth:`~matcal.core.adaptive_surrogates.AdaptiveSurrogateStudyBase.set_number_of_test_samples`
        (default is ``max_training_samples // 20`` or ``n_parameters * 10``,
        whichever is larger).  Supplying an explicit test data set overrides that
        behavior.

        :param study_results: The test data to be used for surrogate evaluation.
            * **StudyResults** – a :class:`~matcal.core.study_base.StudyResults`
            instance containing the desired parameter history and simulation
            results.
            * **str** – a path to a serialized ``.joblib`` file that, when loaded
            returns a ``StudyResults`` object.
        :type study_results: :class:`~matcal.core.study_base.StudyResults` or ``str``

        :raises TypeError: If ``study_results`` is neither a ``StudyResults`` instance
            nor a string.
        :raises FileNotFoundError: If ``study_results`` is a string but the file
            cannot be located or loaded.
        :raises RuntimeError: If the loaded object is not a ``StudyResults`` instance.

        :notes:
            * The supplied test set is **only** used for validation of the surrogate;
            it is never incorporated into the training data.
            * Calling this method multiple times replaces any previously stored test
            data with the most recent value.
            * The test data must be compatible with the study’s parameter space
            (same parameter names and bounds as the training data).
        """
        check_item_is_correct_type(study_results, (StudyResults, str), "study_results")
        if isinstance(study_results, str):
            self._user_test_data = matcal_load(study_results)
            if not isinstance(self._user_test_data, StudyResults):
                raise RuntimeError(f"The data loaded by loading {study_results} is not " +
                               "a study results object and cannot be used for surrogate testing.")
        elif isinstance(study_results, StudyResults):
            self._user_test_data = study_results
        else:
            raise RuntimeError("Improper study results passed for the " +
                               "adaptive surrogate test data.")

    def _set_default_number_of_test_samples(self) -> int:
        """
        Compute the default number of test samples.

        The rule follows:
        * ``max_training_samples // 20`` (integer division) **or**
        * ``n_parameters * 10``

        Whichever of the two values is larger becomes the default.

        :returns: Default number of test samples for the current study.
        :rtype: int
        """
        candidate_a = int(self._max_training_samples // 20)  # floor division → int
        candidate_b = self._number_parameters * 10
        return max(candidate_a, candidate_b)

    def set_target_field_name(self, target_field_name):
        """Specify the field name for the response that the surrogate model 
        will seek to replicate. This is generally a model response such as temperature, 
        load, etc.

        :param target_field_name: the name of the field that the surrogate will 
            replicate
        :type target_field_name: str
        """
        check_value_is_nonempty_str(target_field_name, "target_field_name")
        self._target_field_name = target_field_name

    def _get_test_data(self):
        results = None
        if self._user_test_data is not None:
            results = self._user_test_data
        else:
            results = self._run_test_sampling()
        test_params = self._format_params(results)
        test_responses = self._format_output(results)
        return test_params, test_responses

    def _run_test_sampling(self):
        """
        Execute the initial test‑sampling phase in a dedicated sub‑directory.

        This method:

        1. Sets up a temporary working directory via
           :meth:`_setup_test_sampling_working_directory`.
        2. Calls the parent ``HaltonStudy`` ``launch`` method to generate the
           test samples.
        3. Restores the original working directory.
        4. Stores the formatted test parameters and responses for later use.

        :raises RuntimeError: If the test‑sampling launch fails.
        """
        orig_remove = self._remove_existing_working_directory
        orig_working_directory = self._update_work_dir_for_test_sampling()
        seed = self._seed
        if self._test_group_random_seed is not None:
            self.set_seed(self._test_group_random_seed)
        super().launch()
        test_results = copy.deepcopy(self._results)
        self._reset_study_after_test_sampling_generation(orig_working_directory, orig_remove)
        if seed is not None:
            self.set_seed(seed)
        return test_results

    def set_seed(self, seed):
        """
        Set the study seed and synchronize the Voronoi sampler RNG.

        This keeps K-fold assignment, K-means grouping, and random candidate
        selection tied to the study seed.
        """
        super().set_seed(seed)
        self._random_generator = np.random.default_rng(seed)

    def set_test_group_random_seed(self, seed):
        """
        Set the random seed for the random generator that the study uses
        to generate the  test samples only.
        The method should be called **before**
        :meth:`launch` to
        guarantee reproducibility.

        :param seed: Integer seed for the pseudo‑random number generator.
        :type seed: int
        """
        check_value_is_positive_integer(seed, "seed")
        self._test_group_random_seed = seed

    def _update_work_dir_for_test_sampling(self):
        """
        Prepare a temporary working directory for the initial test‑sampling phase.

        The method follows the same logic that was previously in
        :meth:`launch`:

        * If the user has already set a working directory (``self._working_directory``)
          it is suffixed with ``"_test_samples"``.
        * If no working directory was set, a new directory named ``"test_samples"``
          is used.

        The original directory (or ``None`` if none was set) is returned so the
        caller can restore it after the test run has completed.

        :return: The original working‑directory path before it was modified,
                 or ``None`` if the study had no working directory set.
        :rtype: str | None
        """
        original_dir = None
        if self._working_directory is not None:
            original_dir = self._working_directory
            test_samples_directory = self._working_directory + "_test_samples"
        else:
            test_samples_directory = os.path.abspath("test_samples")
        self.set_working_directory(test_samples_directory, 
                                   remove_existing=self._remove_existing_working_directory)
        return original_dir

    def _reset_study_after_test_sampling_generation(self, orig_working_directory, 
                                                    remove_existing):
        self._working_directory = orig_working_directory
        self._remove_existing_working_directory = remove_existing
        self._results = None
        self._next_evaluation_id_number = 1

    def add_evaluation_set(self, model, state=None, qoi_extractor=None):
        """
        Add an evaluation set that uses a 
        :class:`~matcal.core.objective.SimulationResultsSynchronizer`
        generated from the study’s independent variable, independent‑variable values,
        and target field name.

        .. warning::
            For adaptive surrogates, this can only be called **once** as the
            training points are adaptively chosen based on the response of
            interest.

        This method is a thin wrapper around :meth:`StudyBase.add_evaluation_set`.  
        It accepts only the *model* (required) and an optional *state* argument.
        ``state`` must be a single :class:`~matcal.core.state.State` instance; a
        collection of states is **not** supported.  If ``state`` is ``None`` MatCal's
        default state will be used.
        
        The synchronizer is built automatically from the attributes that
        were defined via :meth:`set_independent_variable` and
        :meth:`set_target_field_name`.

        :param model: The model that will generate the simulation data.
        :type model: :class:`~matcal.core.models.ModelBase`
        :param state: The single state to which the evaluation set should be applied.
            If ``None`` the model’s default state is used.
        :type state: :class:`~matcal.core.state.State`, optional

        :param qoi_extractor: Provide a  
            :class:`~matcal.core.qoi_extractor.UserDefinedExtractor` that will act on the 
            simulation results to provide a quantity of interest for the surrogate.
            It must return target field values of the same length of the 
            independent variable values.
        :type qoi_extractor: :class:`~matcal.core.qoi_extractor.UserDefinedExtractor`
        
        :raises RuntimeError: If the required attributes for the synchronizer
            (independent variable, its values, or target field name) have not been set.
        """
        if self._evaluation_set_added:
            raise RuntimeError(
                "add_evaluation_set can only be called once for a "
                f"{self.__class__.__name__} instance because adaptivity "
                "is only supported for a single model and single response of interest."
            )
        

        if state is not None and not isinstance(state, State):
            raise TypeError(
                f"{self.__class__.__name__}.add_evaluation_set expects ``state`` "
                "to be a single `State` instance (or None)."
            )

        self._results_synchronizer = self._make_simulation_results_synchronizer(qoi_extractor)
        super().add_evaluation_set(
            model,
            objectives=self._results_synchronizer,
            data=None,
            states=state,
        )
        self._evaluation_set_added = True

    def _make_simulation_results_synchronizer(self, qoi_extractor):
        """
        Build a :class:`~matcal.core.objective.SimulationResultsSynchronizer`
        that will be used by the surrogate study.

        The synchronizer evaluates the *simulation* at the user‑provided
        independent‑variable locations and extracts the target field as the
        dependent quantity.

        :return: Configured ``SimulationResultsSynchronizer`` instance.
        :rtype: SimulationResultsSynchronizer
        :raises RuntimeError: If any of the required attributes have not been set.
        """
        if self._independent_variable is None:
            raise RuntimeError(
                "Independent variable name has not been set. Call "
                "`set_independent_variable` before creating the synchronizer."
            )

        if self._target_field_name is None:
            raise RuntimeError(
                "Target field name has not been set. Call "
                "`set_target_field_name` before creating the synchronizer."
            )
        sim_res_synch = SimulationResultsSynchronizer(
            self._independent_variable, self._independent_variable_values,
            self._target_field_name          
        )

        if qoi_extractor is not None:
            if not isinstance(qoi_extractor, UserDefinedExtractor):
                raise TypeError(f"The qoi extractor passed to {self.__class__.__name__} "+
                                f"must be a UserDefinedExtractor. Received "+
                                f"argument of type '{type(qoi_extractor)}'. Check input.")
            sim_res_synch.set_simulation_qoi_extractor(qoi_extractor)
        return sim_res_synch

    def launch(self):
        """
        Run the initial test-sampling study in a dedicated sub-directory,
        then continue with the adaptive surrogate workflow.

        The test-sampling phase is performed by a standard HaltonStudy to generate
        the required test points unless user-provided test data has been supplied.
        """
        test_params, test_responses = self._get_test_data()
        param_names = self._parameter_collection.get_item_names()

        self._surrogate = self._adaptive_surrogate_class(
            self._target_field_name,
            self._independent_variable,
            self._independent_variable_values,
            test_params,
            test_responses,
            param_names,
            self._bounds,
            storage_best_n_surrogates=self._surrogate_storage_best_n_surrogates,
            storage_every_n_batches=self._surrogate_storage_every_n_batches,
            storage_score_metric=self._surrogate_storage_score_metric,
        )

        self._run_study = self._perform_adaptive_surrogate_batch_sampling
        return super().launch()

    def _stopping_criterion_met(self, training_batch_number, stop=False):
        if training_batch_number > 0:
            if np.abs(self._surrogate.rmse_history[-1]) <= self._rmse_goal:
                logger.info(f"Root mean squared error converged!")
                stop=True
            elif np.abs(self._surrogate.max_error_history[-1]) <=self._max_abs_error_goal:
                logger.info(f"Max absolute error score converged!")
                stop=True
        if self._results.number_of_evaluations > self._max_training_samples and not stop:
            logger.info("Surrogate not converged yet, but maximum training "+
                        "samples reached. Exiting.")
            stop=True
        if stop:
            logger.info(f"Surrogate trained on {self._results.number_of_evaluations} samples.")
        else:
            logger.info("Surrogate not converged yet.")
        logger.info(f"Root mean squared error: {self._surrogate.rmse_history[-1]}")
        logger.info(f"Max error score: {self._surrogate.max_error_history[-1]}")
        logger.info(f"R2 score: {self._surrogate.score()}\n")
        return stop
        
    def _matcal_evaluate_parameter_sets_batch_adaptive_training(self, parameter_sets):
        self._populate_parameter_evaluations_adaptive(parameter_sets)
        current_batch = len(self._surrogate.sample_count_history)
        logger.info(f"Active learning batch {current_batch+1}. ")
        if current_batch > 0:
            logger.info(f"Currently the surrogate is trained on "+
                        f"{self._surrogate.sample_count_history[-1]} samples.")
        logger.info("................................................................")
        eval_meth = super()._matcal_evaluate_parameter_sets_batch
        batch_results = eval_meth(self._parameter_sets_to_evaluate)
        return self._format_batch_results(batch_results, parameter_sets)

    def _add_parameter_evaluation(self, **p):
      super()._add_parameter_evaluation(**p)

    @property
    def surrogate(self):
        """
        Return the :class:`~matcal.core.adaptive_surrogates.AdaptiveSurrogate` 
        instance that holds the
        surrogate models and their training history.

        :return: The surrogate object, or ``None`` if the study has not yet
            created one (i.e., before :meth:`launch` is called).
        :rtype: :class:`~matcal.core.adaptive_surrogates.AdaptiveSurrogate` | None
        """
        return self._surrogate
    
    def set_surrogate_save_filename(self, filename):
        """
        Set the path used to save the surrogate object after each training batch.

        The surrogate (an :class:`~matcal.core.adaptive_surrogates.AdaptiveSurrogate` 
        instance) is periodically saved to
        disk with :func:`matcal.core.serializer_wrapper.matcal_save`.  The filename
        must be a non‑empty string that ends with the ``.joblib`` extension.
        The directory
        component of the path is not created automatically; it must already exist
        or be created by the user prior to calling this method.

        :param str filename: Full path (absolute or relative) to the file that will
            store the surrogate.  The filename **must** end with ``.joblib``.
            Example: ``"my_model_sparse_grid_surrogate.joblib"`` or
            ``"/tmp/surrogate.joblib"``.

        :raises ValueError: If *filename* does not contain the required ``.joblib``
            suffix or is empty.
        :raises TypeError: If *filename* is not a string.
        """
        check_value_is_nonempty_str(filename, "filename")
        if not filename.endswith(".joblib"):
            raise ValueError("The save filename for the Adaptive Surrogate Study " +
                f"must end with \".joblib\". Passed filename is \"{filename}\".")
        self._surrogate_save_filename = filename

    @property
    def surrogate_save_filename(self):
        """
        Retrieve the filename (including the ``.joblib`` extension) that will be
        used to save the surrogate object after each training batch.

        :return: The absolute or relative path supplied via
            :meth:`set_surrogate_save_filename`, or ``None`` if no filename has been set.
        :rtype: str | None
        """
        return self._surrogate_save_filename

    @property
    def results_synchronizer(self):
        """
        Return the :class:`~matcal.core.objective.SimulationResultsSynchronizer` that
        was created for this adaptive surrogate study.

        The synchronizer is responsible for evaluating the model at the user‑provided
        independent‑variable locations and extracting the target field (the quantity of
        interest) from the simulation output.  It is constructed the first time
        :meth:`add_evaluation_set` is called and stored internally as
        ``self._results_synchronizer``. As a result, this should be called after 
        an evaluation set is added to the study.

        :return: The ``SimulationResultsSynchronizer`` instance associated with the
            study, or ``None`` if the synchronizer has not yet been created (i.e.
            ``add_evaluation_set`` has not been called).
        :rtype: :class:`~matcal.core.objective.SimulationResultsSynchronizer` | None
        """
        return self._results_synchronizer

    def _format_params(self, results):
        params_formatted = []
        for param in results.parameter_history:
            params_formatted.append(results.parameter_history[param])
        return np.array(params_formatted).T

    def _format_output(self, results):
        model_name = self._get_model_names()[0]
        objective = self._results_synchronizer
        state_name = results.simulation_history[model_name].state_names[0]
        qois = results.qoi_history[f"{model_name}:{objective.name}"]
        sim_qois = qois.simulation_qois
        nsamples = results.number_of_evaluations
        nqois = len(self._independent_variable_values)
        data = np.zeros((nsamples, nqois))
        for idx, sim_qoi in enumerate(sim_qois):
            data[idx,:] = sim_qoi[state_name][0][self._target_field_name]
        return data

    def set_surrogate_storage_options(self, best_n_surrogates=1,
                                      save_every_n_batches=None,
                                      score_metric="max_error"):
        """
        Configure how many trained surrogate objects are retained in the saved
        :class:`AdaptiveSurrogate`.

        All score histories, test parameters, and test responses are always
        stored. This method only controls which surrogate model objects are
        retained.

        :param best_n_surrogates: Retain the best N surrogate objects according
            to ``score_metric``. Set to ``None`` to disable score-based retention.
        :type best_n_surrogates: int or None

        :param save_every_n_batches: Retain every N-th batch surrogate in
            addition to score-based retained surrogates. For example,
            ``save_every_n_batches=5`` retains batches 5, 10, 15, ...
        :type save_every_n_batches: int or None

        :param score_metric: Metric used to define the best surrogates. Supported
            values are ``"rmse"``, ``"max_error"``, ``"r2"``, and ``"score"``.
            ``"score"`` is treated as an alias for ``"r2"``.
        :type score_metric: str
        """
        check_value_is_positive_integer_or_none(best_n_surrogates, "best_n_surrogates")
        check_value_is_positive_integer_or_none(save_every_n_batches, "save_every_n_batches")
        check_value_is_nonempty_str(score_metric, "score_metric")

        score_metric = score_metric.lower().strip()
        valid_metrics = ("rmse", "max_error", "r2", "score")
        if score_metric not in valid_metrics:
            raise ValueError(
                f"score_metric must be one of {valid_metrics}. "
                f"Received '{score_metric}'."
            )

        if best_n_surrogates is None and save_every_n_batches is None:
            raise ValueError(
                "At least one surrogate retention option must be active."
            )

        self._surrogate_storage_best_n_surrogates = best_n_surrogates
        self._surrogate_storage_every_n_batches = save_every_n_batches
        self._surrogate_storage_score_metric = score_metric


class SparseGridAdaptiveSurrogateStudy(AdaptiveSurrogateStudyBase):
    """
    Build an adaptive sparse-grid surrogate using PyApprox's *fitter/result* API
    (``SingleFidelityAdaptiveSparseGridFitter``).

    This study supports two basis families:

    * Global Lagrange basis on nested Clenshaw-Curtis rules (``basis_type="lagrange"``).
      Best for smooth responses; may exhibit oscillations for kinks/discontinuities.

    * Local piecewise polynomial basis (``basis_type="piecewise"``) with degree
      1 (linear), 2 (quadratic), or 3 (cubic). More stable for non-smooth responses.

    Use :meth:`set_sparse_grid_basis` to choose the basis.
    
    These generally behave well for larger parameter spaces.
    Some downsides for these surrogates
    is that one must be trained independently for each response of interest. 
    As a result, this surrogate requires only a single model and state be passed to it.
    It also requires that a target field name be specified for building the surrogate that 
    signifies the response of interest for the surrogate.
    """

    _adaptive_surrogate_class = SparseGridAdaptiveSurrogate

    def __init__(self, *parameters):
        super().__init__(*parameters)
        self._sg_basis_type = "lagrange"   # "lagrange" or "piecewise"
        self._sg_piecewise_degree = 2      # 1,2,3 (only used if piecewise)
        self._sg_max_level = 20
        self._sg_pnorm = 1.0

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

        # Run at least one refinement step, then check the existing criteria.
        while True:
            new_samples = fitter.step_samples()
            if new_samples is None:
                logger.info("No more admissible sparse-grid indices. Stopping.")
                break

            # new_samples are physical/native parameter values with shape
            # (nvars, nsamples_new).
            new_vals = self._matcal_evaluate_parameter_sets_batch_adaptive_training(new_samples)

            # new_vals comes back as (nsamples_new, n_qois); fitter wants
            # (n_qois, nsamples_new).
            if new_vals.ndim != 2 or new_vals.shape[1] != n_qois:
                raise RuntimeError(
                    "Batch evaluation must return array with shape (nsamples, n_qois). "
                    f"Got {new_vals.shape}"
                )

            fitter.step_values(new_vals.T)

            # Store this iteration's surrogate.
            result = fitter.result()
            self._surrogate._add_iteration(result, self._results.number_of_evaluations)

            # Persist after each batch.
            matcal_save(self._surrogate_save_filename, self._surrogate)

            training_batch_number = len(self._surrogate.sample_count_history)
            if self._stopping_criterion_met(training_batch_number):
                break

        return self._results

    def _populate_parameter_evaluations_adaptive(self, samples):
        samples = np.asarray(samples, dtype=float)
        super()._populate_parameter_evaluations(samples.T)

    def _format_batch_results(self, batch_results, parameter_sets_from_pyapprox):
        model_name = self._get_model_names()[0]
        objective_name = self._results_synchronizer.name
        state_name = self._results.simulation_history[model_name].state_names[0]
        formatted_results = np.zeros((parameter_sets_from_pyapprox.shape[1],
                                      len(self._independent_variable_values)))
        for idx, qoi in enumerate(batch_results["qois"]):
            qoi = qoi[model_name][objective_name]
            formatted_results[idx, :] = qoi.simulation_qois[state_name][0][self._target_field_name]
        return formatted_results

    def set_sparse_grid_basis(self, basis_type="lagrange", piecewise_degree=2):
        """
        Select the 1-D basis used by the PyApprox adaptive sparse grid.

        :param basis_type: Either ``"lagrange"`` (global Lagrange on nested Clenshaw-Curtis nodes)
            or ``"piecewise"`` (local piecewise polynomials).
        :type basis_type: str

        :param piecewise_degree: Polynomial degree for the piecewise basis.
            Only used when ``basis_type="piecewise"``.
            Must be 1 (linear), 2 (quadratic), or 3 (cubic).
        :type piecewise_degree: int
        """
        check_value_is_nonempty_str(basis_type, "basis_type")
        basis_type = basis_type.lower().strip()
        if basis_type not in ("lagrange", "piecewise"):
            raise ValueError("basis_type must be 'lagrange' or 'piecewise'")
        self._sg_basis_type = basis_type
        if basis_type == "piecewise":
            check_value_is_positive_integer(piecewise_degree, "piecewise_degree")
            if piecewise_degree not in (1, 2, 3):
                raise ValueError("piecewise_degree must be 1, 2, or 3")
            self._sg_piecewise_degree = int(piecewise_degree)

    def set_sparse_grid_adaptivity_limits(self, max_level=20, pnorm=1.0):
        """
        Set admissibility limits for the adaptive sparse-grid index set.

        This controls which multi-indices are considered admissible by PyApprox's
        greedy refinement algorithm.

        :param max_level: Maximum total level (in the specified p-norm) allowed.
        :type max_level: int

        :param pnorm: The p-norm used by the admissibility criteria. Common is 1.0.
        :type pnorm: float
        """
        check_value_is_positive_integer(max_level, "max_level")
        check_value_is_positive_real(pnorm, "pnorm")
        self._sg_max_level = int(max_level)
        self._sg_pnorm = float(pnorm)


def _fit_surrogate_model(eval_info, interpolation_field, interpolation_locations, 
                         test_eval_info, target_field, save_filename='voronoi_surrogate',  
                         logger_on=True, **kwargs):
    from matcal.core.surrogates import SurrogateGenerator
    decomp_var=0.99
    if "decomp_var" in kwargs:
        decomp_var = kwargs.pop("decomp_var")
    surrogate_generator = SurrogateGenerator(eval_info, training_fraction=1.0, 
                                             interpolation_field=interpolation_field, 
                                             interpolation_locations=interpolation_locations, 
                                             test_eval_info=test_eval_info, **kwargs)
    surrogate_generator.set_fields_of_interest(target_field)
    surrogate_generator.set_PCA_details(decomp_var=decomp_var)
    surrogate_generator._logger_on=logger_on
    save_filename = save_filename.split(".joblib")[0]
    return surrogate_generator.generate(save_filename)
        

class VoronoiAdaptiveSurrogateStudy(AdaptiveSurrogateStudyBase):

    _adaptive_surrogate_class = AdaptiveSurrogate

    def _return_none(*args, **kwargs):
        return None
    
    def __init__(self, *parameters):
        """
        Initialize the Voronoi adaptive surrogate study.

        This initializes sampling options, cross-validation options, convergence
        options, surrogate score histories, and the reproducible random generator
        used by the adaptive sampler.
        """
        super().__init__(*parameters)

        self._num_initial_samples = None
        self._test_eval_info = None
        self.set_number_of_initial_samples()

        self._voronoi_type = None
        self._finite_only = None
        self._iterative_updates = None
        self._thin = None
        self._random_selection = None
        self.set_voronoi_sampling_options()

        self._nsplits = None
        self._nmax_folds = None
        self._nmax_loo = None
        self._batch_size = None
        self._cv_scale = None
        self._cv_metric = None
        self._group_kfold = None
        self._loo_errors = None
        self.set_cross_validation_options()

        self._nbatch_samples = []
        self.test_eval_info = None

        self._convergence_metric = None
        self._eps = None
        self.set_convergence_criteria()

        self._current_surrogate_score = {"score": [], "nlpd": [], "rmse": []}
        self._max_fold_error_indices = None
        self._surrogate_options = {}

        self._random_generator = np.random.default_rng(getattr(self, "_seed", None))
            
    def _update_surrogate_score(self, surrogate=None):
        """
        Store the latent-space scores for the surrogate produced by the current
        batch.

        This must use the current candidate surrogate, not necessarily
        ``current_surrogate``, because the storage policy may choose not to
        retain the current candidate.
        """
        if surrogate is None:
            surrogate = self._surrogate.current_surrogate

        latent_score = surrogate._latent_scores['test']
        self._current_surrogate_score['score'].append(_get_surrogate_metric(latent_score, 'score'))
        self._current_surrogate_score['nlpd'].append(_get_surrogate_metric(latent_score, 'nlpd'))
        self._current_surrogate_score['rmse'].append(_get_surrogate_metric(latent_score, 'rmse'))
         
    def _build_boundary_hull(self):
        from scipy.spatial import ConvexHull, Delaunay
        self._boundary_points = self._make_nd_grid(2)
        self._boundary_hull = ConvexHull(self._boundary_points)
        self._boundary_hull_eq = self._boundary_hull.equations # (nfacet, ndim + 1)
        self._boundary_hull_V, self._boundary_hull_b = self._boundary_hull_eq[:, :-1],\
            self._boundary_hull_eq[:, -1] # normal, offset
        self._bhullD = Delaunay(self._boundary_points)

    def _make_nd_grid(self, npts_along_dim):
        grid_pts = []
        for param_index in np.arange(self._number_parameters):
            grid_pts.append(np.linspace(self._bounds[param_index][0], self._bounds[param_index][1],
                                        npts_along_dim))
        coords = np.meshgrid(*grid_pts)
        coords_ravel = [np.asarray(coords[i]).ravel() for i in np.arange(self._number_parameters)]
        return np.vstack(tuple(coords_ravel)).T

    def set_number_of_initial_samples(self, num_initial_samples=None):
        """
        :param num_initial_samples: The number of samples to initiate the algorithm with.
        The initial samples are used to train the initial surrogate and built the 
        initial voronoi tessellation. Default 10*ndim.
        :type initial_training_length: None or int
        """
        if num_initial_samples is None:
            self._num_initial_samples = 10*self._number_parameters
        else:
            check_value_is_positive_integer(num_initial_samples, "num_initial_samples")
            self._num_initial_samples = num_initial_samples

    def set_batch_size(self, batch_size=None):
        """
        Set the number of new Voronoi sample locations requested per batch.

        If ``batch_size`` is ``None``, the legacy behavior is preserved by using
        ``nmax_loo`` when it is an integer. If ``nmax_loo='all'``, the default
        batch size is one.
        """
        if batch_size is None:
            if isinstance(self._nmax_loo, (int, np.integer)):
                self._batch_size = int(self._nmax_loo)
            else:
                self._batch_size = 1
            return

        check_value_is_positive_integer(batch_size, "batch_size")
        self._batch_size = int(batch_size)

    def set_voronoi_sampling_options(
        self,
        voronoi_type='full', 
        finite_only=False, 
        iterative_updates = True, 
        thin=None,
        random_selection=None
    ):
        """Set options pertaining to the voronoi sampling algorithm. Properties
        that can be altered are listed below.
        
        :param vornoi_type: Defines which Vornoi-based sampling strategy to use.
            Supported options are:
                * 'full': Constructs the full Voronoi tessellation over all points (Default)
                * 'local': Constructs a local Voronoi tessellation using only nearby
                    points determined by k-nearest neighbors. This can reduce computational
                    cost in high dimensions.
        :type voronoi_type: str
        
        :param finite_only: If True, only Vornoi vertices that lie inside the 
            convex hull defined by the boundary points are consided as candidate sample
            locations. If False, all vertices are considered, and those lying outside
            the parameter bounds are clipped back to the convex hull. This is more flexible but
            can be more computationally expensive, especially in high dimensions. 
        :type finite_onlye: bool
        
        :param iterative_updates: If True, the Voronoi tessellation is recomputed 
            after each new sample is added, promoting a more space-filling design.
            If False, the tessellation is updated once per batch after all samples 
            in the batch are selected. This can be faster but may result in sample 
            clustering. 
        :type iterative_updates: bool

        :param thin: If specified, every nth candidate sample location is selected as a new
            sample location. This can significantly reduce computational 
            cost in high-dimensional spaces.
        :type thin: int or None
        
        :param random_selection: If sepecified, this defines the number of candidate sample
            locations that are randomly selected as new samples. This provides an 
            alternative way to reduce computational cost in high-dimensional problems. 
        :type random_selection: int or None
        """
        self._voronoi_type = self._validate_voronoi_type(voronoi_type)

        check_value_is_bool(finite_only, "finite_only")
        check_value_is_bool(iterative_updates, "iterative_updates")

        self._finite_only = finite_only
        self._iterative_updates = iterative_updates
        self._thin = self._validate_optional_positive_integer(thin, "thin")
        self._random_selection = self._validate_optional_positive_integer(
            random_selection, "random_selection"
        )
        self._raise_if_multiple_candidate_reduction_options_active()
    
    def _validate_voronoi_type(self, voronoi_type):
        """
        Validate and normalize the Voronoi tessellation mode.
        """
        check_value_is_nonempty_str(voronoi_type, "voronoi_type")
        voronoi_type = voronoi_type.lower().strip()

        if voronoi_type not in ("full", "local"):
            raise ValueError(
                "Voronoi type must be either 'full' or 'local', "
                f"received '{voronoi_type}'."
            )
        return voronoi_type

    def _validate_optional_positive_integer(self, value, name):
        """
        Validate an optional positive integer setting.
        """
        if value is None:
            return None

        check_value_is_positive_integer(value, name)
        return int(value)

    def _raise_if_multiple_candidate_reduction_options_active(self):
        """
        Reject simultaneous use of ``thin`` and ``random_selection``.
        """
        if self._thin is not None and self._random_selection is not None:
            raise ValueError(
                "Only one of 'thin' and 'random_selection' can be activated. "
                "Not both."
            )

    def set_surrogate_options(self, **kwargs):
        """
        :param regressor_kwargs: A keyword selection of parameters to pass to the predictor used. 
            Please refer to the sklearn documentation for more information for what can be passed to 
            the predictors. 
        """
        self._surrogate_options = kwargs

    def set_convergence_criteria(self, eps=1e-12, convergence_metric='nlpd'):
        """
        Convergence is determined by comparing RMSE or NLPD of
        surrogate between two successive batches.

        :param convergence_metric: Choose from root mean squared error ('rmse') 
            or negative log posterior density ('nlpd') to track surrogate performance
            at each batch iteration. This metric is used to determine if the surrogate
            has converged according to eps. 
        :type convergence metric: str
        
        :param eps: Tolerance for surrogate convergence. 
        :type eps: float 
        """
        self._eps = eps
        self._convergence_metric = convergence_metric

    def set_cross_validation_options(self, nsplits=10, nmax_folds=3, nmax_loo=10, cv_scale=1.0,
                                     cv_metric='sum_abs', group_kfold=False, batch_size=None):
        """
        Configure the cross-validation options used to select Voronoi refinement
        regions.

        The Voronoi adaptive sampler can use a two-stage error filter modeled after
        the KFCV-Voronoi procedure:

        1. Split the current training samples into ``nsplits`` folds and build one
        surrogate per held-out fold.
        2. Compute the physical response-space prediction error on each held-out
        fold.
        3. Select the ``nmax_folds`` folds with the largest physical
        cross-validation errors.
        4. Optionally perform leave-one-out cross validation only on the samples
        contained in those selected folds.
        5. Select Voronoi cells associated with the largest leave-one-out physical
        errors and place new samples at farthest vertices of those cells.

        If ``nsplits`` is set to ``0``, the cross-validation filter is disabled.
        In that case, all current training samples are treated as candidate Voronoi
        cell seeds.

        Cross-validation and leave-one-out errors are computed in physical response
        space by comparing held-out model responses with surrogate predictions at
        the same held-out parameter locations. They are not based on latent-space
        surrogate diagnostics.

        :param nsplits: Number of folds used for K-fold cross validation. If
            ``nsplits=0``, K-fold cross validation and leave-one-out cross
            validation are skipped and candidate Voronoi regions are drawn from all
            current training samples. If ``nsplits`` is larger than the current
            number of training samples, it is reduced internally.
        :type nsplits: int

        :param nmax_folds: Number of highest-error K-folds retained for possible
            refinement. Samples contained in these folds define the candidate
            regions for leave-one-out cross validation. This corresponds to the
            number of K-fold groups selected by the global KFCV filter.
        :type nmax_folds: int

        :param nmax_loo: Number of highest-error leave-one-out samples retained
            after the K-fold filter. These samples define the Voronoi regions from
            which new adaptive samples are drawn. If ``nmax_loo='all'``,
            leave-one-out cross validation is skipped and all samples in the
            selected high-error folds are used as candidate Voronoi regions.
        :type nmax_loo: int or str

        :param cv_scale: Optional scaling applied to physical responses before
            computing cross-validation errors. Use this to normalize response
            magnitudes when the target response has multiple components or when
            different response locations have substantially different scales.
            Accepted values are:

            * ``None`` or ``1.0``: no scaling;
            * positive scalar: divide all response values by this scalar;
            * positive array-like: divide response values componentwise; or
            * ``"cbrt"``: apply a cube-root transform to true and predicted
            responses before error calculation.

        :type cv_scale: float, array-like, str, or None

        :param cv_metric: Physical response-space error metric used for both K-fold
            and leave-one-out ranking. Supported values are:

            * ``"rmse"``: root mean squared physical response error.
            * ``"mae"`` or ``"abs"``: mean absolute physical response error.
            * ``"sum_abs"``: sum of absolute physical response errors. This is
            closest to the error expression used in the KFCV-Voronoi paper.
            * ``"nrmse"``: normalized root mean squared physical response error.
            * ``"nlpd"``: accepted for backward compatibility. Because physical
            NLPD requires predictive variances, this option is evaluated as
            physical RMSE for adaptive region ranking.

        :type cv_metric: str

        :param group_kfold: If ``True``, samples are grouped using k-means
            clustering before K-fold cross validation and nearby samples are kept in
            the same validation fold using ``GroupKFold``. This can reduce leakage
            between spatially correlated training and validation points. If
            ``False``, samples are assigned to folds with standard shuffled
            ``KFold``.
        :type group_kfold: bool

        :raises TypeError: If input types are invalid.
        :raises ValueError: If numeric options are out of range, if ``nmax_loo`` is
            a string other than ``"all"``, or if ``cv_metric`` is unsupported.

        :notes:
            * ``nmax_folds`` controls how many high-error K-fold groups pass the
            global KFCV filter.
            * ``nmax_loo`` controls how many individual high-error samples are used
            after the optional leave-one-out refinement step.
            * The actual number of new samples added in a batch may be smaller than
            ``nmax_loo`` if some Voronoi regions produce invalid, duplicate, or
            out-of-bounds candidate points.
            * For behavior closest to the paper, use ``cv_metric="sum_abs"``.
            For fold-size-independent ranking, use ``cv_metric="rmse"``.
        """
        check_value_is_nonnegative_integer(nsplits, "nsplits")
        self._nsplits = int(nsplits)

        check_value_is_positive_integer(nmax_folds, "nmax_folds")
        self._nmax_folds = int(nmax_folds)

        self._nmax_loo = self._validate_nmax_loo(nmax_loo)
        self._cv_scale = _validate_cv_scale(cv_scale)
        self._cv_metric = self._validate_cv_metric(cv_metric)

        check_value_is_bool(group_kfold, "group_kfold")
        self._group_kfold = group_kfold

        self.set_batch_size(batch_size)

    def _validate_nmax_loo(self, nmax_loo):
        """
        Validate the number of LOO-ranked candidates retained after KFCV filtering.
        """
        if isinstance(nmax_loo, str):
            nmax_loo = nmax_loo.lower().strip()
            if nmax_loo == "all":
                return nmax_loo
            raise ValueError("If nmax_loo is a string, it must be 'all'.")

        check_value_is_positive_integer(nmax_loo, "nmax_loo")
        return int(nmax_loo)


    def _validate_cv_metric(self, cv_metric):
        """
        Validate and normalize the physical cross-validation error metric.
        """
        check_value_is_nonempty_str(cv_metric, "cv_metric")
        cv_metric = cv_metric.lower().strip()

        valid_cv_metrics = ("rmse", "mae", "abs", "sum_abs", "nrmse", "nlpd")
        if cv_metric not in valid_cv_metrics:
            raise ValueError(
                "cv_metric not implemented. cv_metric must be one of "
                f"{valid_cv_metrics}. Received '{cv_metric}'."
            )

        return cv_metric

    def _format_output_for_surrogate_gen(self, results):
        from matcal.core.data import convert_data_to_dictionary
        model_name = self._get_model_names()[0]
        state_name = results.simulation_history[model_name].state_names[0]
        sim_history = self._results.simulation_history[model_name][state_name]
        nsamples = results.number_of_evaluations
        data = []
        for nn in np.arange(nsamples):
            data.append(convert_data_to_dictionary(sim_history[nn]))
        return data
    
    def _reset_study_after_test_sampling_generation(self, orig_working_directory, remove_existing):
        self._test_eval_info = copy.deepcopy(self._results)
        super()._reset_study_after_test_sampling_generation(orig_working_directory, remove_existing)
        
    def _perform_adaptive_surrogate_batch_sampling(self):
        """
        Run the Voronoi adaptive sampling loop.

        The loop now exits gracefully if a batch produces no valid sample
        locations.
        """
        self._initialize_voronoi_surrogate_run()
        training_params, training_data = self._run_initial_training_samples()
        batch_number = 0

        while not self._stopping_criterion_met(batch_number):
            self._log_voronoi_batch_start(batch_number)
            new_points = self._get_next_voronoi_batch(
                batch_number, training_params, training_data
            )

            if new_points.size == 0:
                logger.warning("No valid Voronoi sample locations found. Stopping.")
                break

            self._evaluate_voronoi_batch(new_points)
            training_params, training_data = self._train_surrogate_with_current_results()
            batch_number += 1

        return self._results

    def _initialize_voronoi_surrogate_run(self):
        """
        Initialize file names, boundary geometry, and parameter-name bookkeeping.
        """
        if self._surrogate_save_filename is None:
            self.set_surrogate_save_filename(
                f"{self._get_model_names()[0]}_voronoi_adaptive_surrogate.joblib"
            )

        self._build_boundary_hull()
        self.param_names = self._parameter_collection.get_item_names()

    def _log_voronoi_batch_start(self, batch_number):
        """
        Log the start of a Voronoi active-learning batch.
        """
        logger.info(
            f"Active learning batch {batch_number + 1}."
            f"\nCurrently the surrogate is trained on {self._nbatch_samples[-1]} samples."
        )
        logger.info("................................................................")

    def _get_next_voronoi_batch(self, batch_number, training_params, training_data):
        """
        Select and validate the next batch of Voronoi sample locations.
        """
        new_points = self._create_voronoi_tess_and_choose_new_samples(
            batch_number, training_params, training_data
        )

        return self._check_points_within_bounds(new_points)

    def _evaluate_voronoi_batch(self, new_points):
        """
        Evaluate the model at a batch of new Voronoi-selected points.
        """
        self._populate_parameter_evaluations(new_points)
        self._matcal_evaluate_parameter_sets_batch(self._parameter_sets_to_evaluate)

    def _stopping_criterion_met(self, training_batch_number, stop=False):
        scores = self._current_surrogate_score
        if training_batch_number > 1:
            this_score = scores[self._convergence_metric][training_batch_number]
            last_score = scores[self._convergence_metric][training_batch_number-1]
            if np.abs(this_score - last_score) <= self._eps:
                logger.info(f"Surrogate Converged!\n"+
                             f"Convergence from surrogate '{self._convergence_metric}' score:")
                logger.info(f"Final score: {this_score}")
                logger.info(f"Score delta: {np.abs(this_score - last_score)}")
                logger.info(f"Score delta convergence criteria: {self._eps}\n")
                
                stop = True
        return super()._stopping_criterion_met(training_batch_number, stop)

    def _run_initial_training_samples(self):
        super().set_number_of_samples(self._num_initial_samples)
        super()._generate_samples(self._num_initial_samples, self._skip)
        self._matcal_evaluate_parameter_sets_batch(self._parameter_sets_to_evaluate)        
        return self._train_surrogate_with_current_results()
    
    def _train_surrogate_with_current_results(self):
        training_params = self._format_params(self._results)
        training_data = self._format_output_for_surrogate_gen(self._results)
        current_surrogate = _fit_surrogate_model(
            self,
            interpolation_field=self._independent_variable, 
            interpolation_locations=self._independent_variable_values, 
            test_eval_info=self._test_eval_info, 
            target_field=self._target_field_name,
            save_filename=self._surrogate_save_filename,
            **self._surrogate_options
        )
        self._surrogate._add_iteration(current_surrogate, self._results.number_of_evaluations)
        self._update_surrogate_score(current_surrogate)
        self._nbatch_samples.append(self.results.number_of_evaluations)
        # Persist the AdaptiveSurrogate container, not only the latest PCA surrogate.
        if self._surrogate_save_filename is not None:
            matcal_save(self._surrogate_save_filename, self._surrogate)
        return training_params, training_data

    def _create_voronoi_tess_and_choose_new_samples(
        self,
        iteration,
        training_params,
        training_data,
    ):
        """
        Select candidate regions, build the Voronoi object, and choose new samples.
        """
        candidates = self._get_voronoi_candidate_locations(training_params, training_data)
        self._worst_sample_locations = self._reduce_candidates(candidates)

        logger.info(f"Initializing voronoi/tree for batch {iteration}")
        self._create_voronoi_tess(training_params)

        return self._find_new_sample_locations()

    def _get_voronoi_candidate_locations(self, training_params, training_data):
        """
        Return candidate sample locations from KFCV/LOO or from all samples.
        """
        if self._nsplits <= 0:
            return training_params

        self._perform_kfold_cross_validation(training_params, training_data)
        self._find_kfold_max_errors()

        if self._nmax_loo == "all":
            candidates = training_params[self._max_fold_error_indices]
            return self._random_subset_rows(candidates, self._batch_size)

        self._perform_loo_cross_validation(training_params, training_data)
        return self._find_loo_max_errors(training_params)

    def _reduce_candidates(self, candidates):
        """
        Apply thinning, random down-selection, and the batch-size limit.
        """
        candidates = self._normalize_candidate_array(candidates)

        if candidates.shape[0] == 0:
            return candidates

        if self._thin is not None:
            candidates = candidates[:: self._thin]

        elif self._random_selection is not None:
            draw_n = min(candidates.shape[0], self._random_selection, self._batch_size)
            candidates = self._random_subset_rows(candidates, draw_n)

        return candidates[: self._batch_size]

    def _normalize_candidate_array(self, candidates):
        """
        Convert candidate locations to a two-dimensional floating-point array.
        """
        candidates = np.asarray(candidates, dtype=float)

        if candidates.size == 0:
            return np.empty((0, self._number_parameters))

        return np.atleast_2d(candidates)

    def _random_subset_rows(self, values, n_rows):
        """
        Select a reproducible random subset of rows using the study RNG.
        """
        values = self._normalize_candidate_array(values)
        n_rows = min(int(n_rows), values.shape[0])

        if n_rows <= 0:
            return np.empty((0, self._number_parameters))

        rows = self._random_generator.choice(values.shape[0], n_rows, replace=False)
        return values[np.sort(rows)]
        
    def _create_voronoi_tess(self, training_params):
        if self._voronoi_type == 'full':
            # Initialize Voronoi tessellation
            self._voronoi_tessellation = VoronoiTessellation(training_params, self._bounds, 
                                                             self._finite_only)
            self._voronoi_tessellation.build()

        elif self._voronoi_type == 'local':
            # Make a local voronoi tesselation for each new sample by using knearest neighbors
            # to determine the closest points
            from scipy.spatial import KDTree
            self._all_tree_points = training_params.copy()
            self._tree = KDTree(self._all_tree_points)
                
    def _find_new_sample_locations(self):
        """
        Find new sample locations from the selected Voronoi candidate regions.

        Local Voronoi sampling now checks that enough nearest neighbors exist before
        attempting to build a local tessellation.
        """
        new_points = []
        logger.info("Finding new sample locations")

        for loc_idx, location in enumerate(self._worst_sample_locations):
            self._log_new_sample_location_progress(loc_idx)
            new_point = self._find_new_sample_location_for_candidate(location)

            if new_point is None:
                continue

            new_points.append(new_point)
            self._update_adaptive_voronoi_after_new_point(new_point)

        return self._package_new_voronoi_points(new_points)

    def _log_new_sample_location_progress(self, loc_idx):
        """
        Log progress while selecting new Voronoi sample locations.
        """
        if np.mod(loc_idx, 100) == 0:
            logger.info(
                f"Drawing new sample from region index {loc_idx} "
                f"of {len(self._worst_sample_locations)}."
            )

    def _find_new_sample_location_for_candidate(self, location):
        """
        Find one new sample location for a selected candidate region.
        """
        try:
            if self._voronoi_type == "full":
                return self._find_full_voronoi_new_point(location)

            if self._voronoi_type == "local":
                return self._find_local_voronoi_new_point(location)

        except (TypeError, ValueError, RuntimeError) as err:
            logger.warning(f"Skipping invalid Voronoi candidate: {err}")
            return None

        raise RuntimeError(f"Unsupported Voronoi type '{self._voronoi_type}'.")

    def _find_full_voronoi_new_point(self, location):
        """
        Find the furthest valid vertex in the full Voronoi tessellation.
        """
        region_index = self._voronoi_tessellation.get_voronoi_region(location)[0][0]
        vertices, furthest_vertex_index = (
            self._voronoi_tessellation.find_furthest_vertex(region_index)
        )

        return self._select_furthest_vertex(vertices, furthest_vertex_index)

    def _find_local_voronoi_new_point(self, location):
        """
        Build a local Voronoi tessellation and select its furthest valid vertex.
        """
        neighbor_points = self._get_local_voronoi_neighbor_points(location)
        local_voronoi = self._build_local_voronoi_tessellation(neighbor_points)

        region_index = local_voronoi.get_voronoi_region(location)[0][0]
        vertices, furthest_vertex_index = local_voronoi.find_furthest_vertex(region_index)

        return self._select_furthest_vertex(vertices, furthest_vertex_index)

    def _select_furthest_vertex(self, vertices, furthest_vertex_index):
        """
        Return a furthest Voronoi vertex or ``None`` when no valid vertex exists.
        """
        if vertices is None or furthest_vertex_index is None:
            return None

        return vertices[furthest_vertex_index]

    def _get_local_voronoi_neighbor_points(self, location):
        """
        Return nearest-neighbor points for a local Voronoi tessellation.

        The neighbor count is clamped so it is never zero and never below the
        minimum needed for a dimensionally valid local tessellation.
        """
        neighbor_count = self._get_local_voronoi_neighbor_count()
        nearest_indices = self._query_local_voronoi_neighbors(location, neighbor_count)

        return self._all_tree_points[nearest_indices]

    def _get_local_voronoi_neighbor_count(self):
        """
        Return a safe neighbor count for local Voronoi construction.
        """
        n_available = self._all_tree_points.shape[0]
        min_neighbors = self._minimum_local_voronoi_neighbors()

        if n_available < min_neighbors:
            raise RuntimeError(
                "Not enough points are available to build a local Voronoi "
                f"tessellation. Need at least {min_neighbors}, but only "
                f"{n_available} are available."
            )

        requested_neighbors = int(np.ceil(0.25 * n_available))
        neighbor_count = max(min_neighbors, requested_neighbors)

        return min(neighbor_count, n_available)

    def _minimum_local_voronoi_neighbors(self):
        """
        Return the minimum number of neighbors for local Voronoi construction.
        """
        return self._number_parameters + 2

    def _query_local_voronoi_neighbors(self, location, neighbor_count):
        """
        Query the KDTree for local Voronoi nearest-neighbor indices.
        """
        _, nearest_indices = self._tree.query(location, k=neighbor_count)
        return np.asarray(nearest_indices, dtype=int).reshape(-1)


    def _build_local_voronoi_tessellation(self, neighbor_points):
        """
        Build and return a local Voronoi tessellation from nearest neighbors.
        """
        local_voronoi = VoronoiTessellation(
            neighbor_points,
            self._bounds,
            self._finite_only,
        )
        local_voronoi.build()

        return local_voronoi

    def _update_adaptive_voronoi_after_new_point(self, new_point):
        """
        Update the active full tessellation or local KDTree after selecting a point.
        """
        if not self._iterative_updates:
            return

        if self._voronoi_type == "full":
            self._voronoi_tessellation.add_points(np.asarray(new_point, dtype=float))
            return

        self._update_local_voronoi_tree(new_point)

    def _update_local_voronoi_tree(self, new_point):
        """
        Add a new point to the local Voronoi KDTree.
        """
        from scipy.spatial import KDTree

        self._all_tree_points = np.vstack(
            (self._all_tree_points, np.atleast_2d(new_point))
        )
        self._tree = KDTree(self._all_tree_points)

    def _package_new_voronoi_points(self, new_points):
        """
        Convert selected Voronoi points to a unique bounded point array.
        """
        if len(new_points) == 0:
            return np.empty((0, self._number_parameters))

        new_points = np.asarray(new_points, dtype=float)
        unique_points = np.unique(new_points, axis=0)

        return self._check_points_within_bounds(unique_points)

    def _check_points_within_bounds(self, points):
        """
        Return finite candidate points that lie inside the parameter bounds.
        """
        points = np.asarray(points, dtype=float)

        if points.size == 0:
            return np.empty((0, self._number_parameters))

        points = np.atleast_2d(points)
        self._check_candidate_point_dimension(points)

        lb = self._bounds[:, 0]
        ub = self._bounds[:, 1]

        mask = np.isfinite(points).all(axis=1)
        mask &= ((points >= lb) & (points <= ub)).all(axis=1)

        return points[mask]

    def _check_candidate_point_dimension(self, points):
        """
        Validate that candidate points have one column per parameter.
        """
        if points.shape[1] != self._number_parameters:
            raise ValueError(
                f"Expected {self._number_parameters} columns, got {points.shape[1]}."
            )   
        
    def _perform_kfold_cross_validation(self, training_params, training_data):
        """
        Perform K-fold CV using a split count valid for the current data size.
        """
        logger.info("Performing kfold cross-validation")

        nsplits = _get_valid_kfold_split_count(self._nsplits, training_params.shape[0])
        kfcv = self._make_kfold_cross_validation_runner(nsplits)
        groups = self._make_kfold_groups(training_params, nsplits)

        self._kf = kfcv.perform_kfold_cv(training_params, training_data, groups)

    def _make_kfold_cross_validation_runner(self, nsplits):
        """
        Construct the K-fold cross-validation helper object.
        """
        return KFoldCrossValidation(
            nsplits,
            self._group_kfold,
            self._independent_variable,
            self._independent_variable_values,
            self._cv_scale,
            self._cv_metric,
            self._target_field_name,
            self.param_names,
            self._surrogate_options,
            random_seed=getattr(self, "_seed", None),
        )

    def _make_kfold_groups(self, training_params, nsplits):
        """
        Build grouping labels for GroupKFold, or return None for standard KFold.
        """
        if not self._group_kfold:
            return None

        from sklearn.cluster import KMeans

        return KMeans(
            n_clusters=nsplits,
            random_state=getattr(self, "_seed", 42),
        ).fit_predict(training_params)

    def _find_kfold_max_errors(self):
        max_folds = self._find_indices_of_n_largest_kf_errors()
        self._max_fold_error_indices = np.concatenate(list(max_folds.values()))
        logger.info(f"\n\tWorst kfold errors associated with the following sample indices:\n"+
            f"\t{self._max_fold_error_indices}\n")
    
    def _perform_loo_cross_validation(self, training_params, training_data):
        self._loo_errors = None
        logger.info("Finding worst sample locations using leave one out validation...")
        loocv = LeaveOneOutCrossValidation(self._cv_scale, self._cv_metric, 
                                           self._independent_variable, 
                                           self._independent_variable_values, 
                                           self._target_field_name,
                                           self.param_names, self._surrogate_options)
        self._loo_errors = loocv.perform_loocv(training_params, training_data,
                                               self._max_fold_error_indices)
    
    def _find_loo_max_errors(self, training_params):
        """
        Return training locations associated with the largest LOO errors.
        """
        self._worst_sample_locations = None
        max_loo_indices = self._find_indices_of_n_largest_errors()

        logger.info(
            "\n\tWorst errors when the following sample indices are left out of "
            f"training:\n\t{max_loo_indices}\n"
        )

        return training_params[max_loo_indices]

    def _find_indices_of_n_largest_errors(self):
        """
        Return sample indices with the largest LOO errors, limited by batch size.
        """
        nkeep = self._get_number_of_loo_errors_to_keep()
        sorted_items = self._sorted_loo_error_items()
        indices = [item[2] for item in sorted_items[:nkeep]]

        return np.asarray(indices[: self._batch_size], dtype=int)

    def _get_number_of_loo_errors_to_keep(self):
        """
        Return the configured number of LOO-ranked candidates to retain.
        """
        if self._nmax_loo == "all":
            return len(self._loo_errors)

        return min(int(self._nmax_loo), len(self._loo_errors))

    def _sorted_loo_error_items(self):
        """
        Return LOO error records sorted from largest error to smallest error.
        """
        items = [(key, value[0], value[1]) for key, value in self._loo_errors.items()]
        return sorted(items, key=lambda x: x[1], reverse=True)
        
    def _find_indices_of_n_largest_kf_errors(self):
        # Create a list of (key, error, sample_index) tuples
        items = [(key, value[0], value[1]) for key, value in self._kf.items()]
        # Sort the items based on the error in descending order
        sorted_items = sorted(items, key=lambda x: x[1], reverse=True)
        # Get the top n items
        top_n_items = sorted_items[:self._nmax_folds]
        # Extract the arrays associated with the top n largest floats
        result_arrays = {key: array for key, _, array in top_n_items}
        return result_arrays

    def _add_parameter_evaluation(self, **p):
      super()._add_parameter_evaluation(**p)

    def add_parameter_evaluation(self, **parameters):
        """"""
        raise self.StudyInputError("Users cannot add parameter evaluations to"
                                   " a VoronoiAdaptiveSurrogateStudy.")


class VoronoiTessellation:
    def __init__(self, points, bounds, finite_only):
        """Initialize the VoronoiBatchSamplingStudy
        
        :param points: Array of points that are the seeds of the Voronoi tessellation
        :type points: nd_array

        :param bounds: Bounds for the parameter space,
            e.g., [(xmin, xmax), (ymin, ymax)] for a 2D space.
        :type bounds: list of tuples
        """
        self.points = np.array(points)
        self.ndim = self.points.shape[1]
        self.bounds = bounds
        self.finite_only = finite_only
        self.incremental = False
        
    def build(self):
        """Initialize the Voronoi tessellation with given points and bounds.
        """
        from scipy.spatial import Voronoi, Delaunay, ConvexHull
        self.boundary_points = self._make_nd_grid(npts_along_dim=2)
        if not self.finite_only:
            self.boundary_hull = ConvexHull(self.boundary_points)
            self.boundary_hull_eq = self.boundary_hull.equations # (nfacet, ndim + 1)
            self.boundary_hull_V, self.boundary_hull_b = \
                self.boundary_hull_eq[:, :-1], self.boundary_hull_eq[:, -1] # normal, offset
            self.bhullD = Delaunay(self.boundary_points)
        else:
            self.boundary_hull = None
            self.bhullD = None
        self.create_ghost_points()
        self._all_points = np.vstack([self.points, self._ghost_points])
        
        self.vor = Voronoi(self._all_points, incremental=self.incremental)
        self.ghost_busters()
        self.boundary_regions = self.get_voronoi_region(self.boundary_points) # may need to update
       
    def _make_nd_grid(self, npts_along_dim):
        grid_pts = []
        for dim in np.arange(self.ndim):
            grid_pts.append(np.linspace(self.bounds[dim,0], self.bounds[dim,1], npts_along_dim))
        coords = np.meshgrid(*grid_pts)
        coords_ravel = [np.asarray(coords[i]).ravel() for i in np.arange(self.ndim)]
        return np.vstack(tuple(coords_ravel)).T
        
    def create_ghost_points(self, stretchCoef=1.75, centCoef=1.5):
        """
        Create auxiliary ``ghost`` seed points outside the bounded parameter domain.

        The Voronoi tessellation is built from the physical training samples plus
        additional ghost points. These ghost points help make the Voronoi regions
        associated with physical samples finite inside the bounded parameter space.
        They are not valid training samples and are used only to stabilize the
        tessellation near the domain boundary.

        Ghost points are generated in two groups:

        1. Boundary-corner ghost points:
        Each corner of the bounded parameter domain is moved outward from the
        domain centroid by ``stretchCoef``. Stretching about the centroid is
        important because it works for both centered domains, such as
        ``[-5, 5]^d``, and noncentered domains, such as ``[0, 1]^d``. Stretching
        about the origin can incorrectly leave ghost points inside the domain.

        2. Axis-direction ghost points:
        Additional points are placed in the positive and negative coordinate
        directions from the domain centroid. These points improve tessellation
        robustness, especially in higher dimensions.

        :param stretchCoef: Multiplicative factor used to move each boundary corner
            away from the domain centroid. Values greater than one place the
            stretched boundary points outside the original domain.
        :type stretchCoef: float

        :param centCoef: Multiplicative factor used with the maximum distance from
            the domain centroid to a boundary corner when placing the additional
            axis-direction ghost points.
        :type centCoef: float

        :ivar _ghost_points: Array of generated ghost points with shape
            ``(n_ghost_points, n_dimensions)``.
        :vartype _ghost_points: numpy.ndarray
        """
        boundary_centroid = np.mean(self.boundary_points, axis=0)

        # Stretch boundary points outward from the domain centroid, not from the
        # origin. This keeps ghost points outside non-origin-centered domains.
        self._ghost_points = (
            boundary_centroid
            + stretchCoef * (self.boundary_points - boundary_centroid)
        )

        max_dist = np.max(
            np.linalg.norm(self.boundary_points - boundary_centroid, axis=1)
        )

        self._ghost_points = np.vstack([
            self._ghost_points,
            boundary_centroid + centCoef * max_dist * np.eye(self.points.shape[1]),
            boundary_centroid - centCoef * max_dist * np.eye(self.points.shape[1]),
        ])

    def ghost_busters(self):
        """ Identify which points in self._all_points are ghost points"""
        self._boo = []
        for point in self._all_points:
            if np.any(np.all(np.isclose(self._ghost_points, point), axis=1)):
                self._boo.append(True)
            else:
                self._boo.append(False)

    def get_region_vertices(self, region_index, identify_outside_vertices=True):
        """
        Return vertices for a Voronoi region.

        Infinite vertices are never used as NumPy indices. Valid vertex index ``0``
        is retained.
        """
        self.raise_if_invalid_region_index(region_index)

        region = self.vor.regions[region_index].copy()

        if not identify_outside_vertices:
            return self._get_unclipped_region_vertices(region)

        return self._get_clipped_region_vertices(region_index, region)

    def _get_unclipped_region_vertices(self, region):
        """
        Return finite region vertices without clipping to the parameter bounds.
        """
        finite_indices = self._finite_vertex_indices(region)
        return self._vertices_from_indices(finite_indices)

    def _finite_vertex_indices(self, region):
        """
        Return valid finite Voronoi vertex indices from a region list.

        SciPy uses ``-1`` to mark infinite vertices. The updated bounded-region
        logic uses ``-2`` for vertices that are infinite or outside the parameter
        bounds. Both are excluded. Valid vertex index ``0`` is retained.
        """
        return [int(idx) for idx in region if int(idx) >= 0]


    def _vertices_from_indices(self, vertex_indices):
        """
        Return Voronoi vertex coordinates for a list of valid finite vertex indices.

        Returns ``None`` if no valid finite vertices are available.
        """
        if len(vertex_indices) == 0:
            return None

        return self.vor.vertices[vertex_indices]

    def _get_clipped_region_vertices(self, region_index, region):
        """
        Return region vertices clipped or filtered to the parameter bounds.

        Finite vertices, clipped ridge-boundary intersections, and associated
        boundary vertices are combined, filtered for finite in-bound coordinates,
        and uniqued.
        """
        updated_region = self.identify_vertices_outside_bounds(region)
        vertices = self._vertices_from_updated_region(region_index, region, updated_region)
        vertices = self._append_boundary_vertices_for_region(region_index, vertices)
        vertices = self._filter_vertices_inside_bounds(vertices)

        if vertices is None or vertices.shape[0] == 0:
            return None

        return np.unique(np.atleast_2d(vertices), axis=0)

    def _vertices_from_updated_region(self, region_index, original_region, updated_region):
        """
        Return valid region vertices from an updated bounded-region list.
        """
        finite_indices = self._finite_vertex_indices(updated_region)

        if -2 not in updated_region:
            return self._vertices_from_indices(finite_indices)

        if self.finite_only:
            return self._vertices_from_indices(finite_indices)

        region_tuple = list(zip(original_region, updated_region))
        return self.replace_unbounded_vertices(updated_region, region_index, region_tuple)

    def _append_boundary_vertices_for_region(self, region_index, vertices):
        """
        Append boundary-corner vertices associated with a bounded Voronoi region.
        """
        pieces = []

        if vertices is not None and len(vertices) > 0:
            pieces.append(np.atleast_2d(vertices))

        if not self.finite_only:
            boundary_vertices = self._get_boundary_vertices_for_region(region_index)
            if boundary_vertices is not None and len(boundary_vertices) > 0:
                pieces.append(np.atleast_2d(boundary_vertices))

        if len(pieces) == 0:
            return None

        return np.concatenate(pieces, axis=0)

    def _get_boundary_vertices_for_region(self, region_index):
        """
        Return parameter-domain boundary points associated with a Voronoi region.
        """
        boundary_indices = [
            idx
            for idx in np.arange(len(self.boundary_regions))
            if self.boundary_regions[idx][0] == region_index
        ]

        if len(boundary_indices) == 0:
            return None

        return self.boundary_points[boundary_indices]

    def _filter_vertices_inside_bounds(self, vertices):
        """
        Keep only finite vertices inside the parameter bounds.
        """
        if vertices is None:
            return None

        vertices = np.atleast_2d(np.asarray(vertices, dtype=float))

        if vertices.size == 0:
            return np.empty((0, self.ndim))

        lb = self.bounds[:, 0]
        ub = self.bounds[:, 1]

        mask = np.isfinite(vertices).all(axis=1)
        mask &= ((vertices >= lb) & (vertices <= ub)).all(axis=1)

        return vertices[mask]

    def get_voronoi_vertices(self, identify_outside_vertices=True):
        """
        Return all valid physical Voronoi vertices.

        Regions belonging to ghost points are skipped.
        """
        vertices = []

        for region_index, _ in enumerate(self.vor.regions):
            if self._region_belongs_to_ghost_point(region_index):
                continue

            region_vertices = self.get_region_vertices(
                region_index,
                identify_outside_vertices,
            )

            if region_vertices is not None and len(region_vertices) > 0:
                vertices.append(np.atleast_2d(region_vertices))

        if len(vertices) == 0:
            return None

        return np.unique(np.concatenate(vertices), axis=0)

    def _region_belongs_to_ghost_point(self, region_index):
        """
        Return True if a Voronoi region belongs to a ghost seed point.
        """
        point_indices = np.where(self.vor.point_region == region_index)[0]

        if len(point_indices) != 1:
            return True

        return self._boo[point_indices[0]]
        
    def identify_vertices_outside_bounds(self, region):
        """
        Mark infinite or out-of-bounds Voronoi vertices as ``-2``.
        """
        region = np.asarray(region, dtype=int)

        if region.size == 0:
            return []

        updated_region = self._mark_infinite_vertices(region)
        updated_region = self._mark_outside_finite_vertices(updated_region)

        return updated_region.tolist()

    def _mark_infinite_vertices(self, region):
        """
        Convert SciPy infinite vertex markers from ``-1`` to ``-2``.
        """
        updated_region = region.copy()
        updated_region[updated_region == -1] = -2
        return updated_region

    def _mark_outside_finite_vertices(self, updated_region):
        """
        Mark finite Voronoi vertices outside the parameter bounds as ``-2``.
        """
        finite_mask = updated_region >= 0

        if not np.any(finite_mask):
            return updated_region

        finite_vertices = self.vor.vertices[updated_region[finite_mask]]
        outside_mask = self._outside_bounds_mask(finite_vertices)

        finite_positions = np.where(finite_mask)[0]
        updated_region[finite_positions[outside_mask]] = -2

        return updated_region

    def _outside_bounds_mask(self, vertices):
        """
        Return a mask indicating which vertices lie outside the parameter bounds.
        """
        outside_mask = np.zeros(vertices.shape[0], dtype=bool)

        for dim in range(self.ndim):
            lb = self.bounds[dim, 0]
            ub = self.bounds[dim, 1]
            outside_mask |= (vertices[:, dim] < lb) | (vertices[:, dim] > ub)

        return outside_mask

    def replace_unbounded_vertices(self, region, region_index, region_tuple):
        """
        Replace the infinite vertices in a Voronoi region with new vertices on 
        the edge of the bounding box.
        ** vertices that sit outside the bounding region are considered infinite here

        :param region: A list of the voronoi regions. Each list contains indices
        of the Voronoi vertices forming each Voronoi region.
        :type region: list
        
        :param region_index: Region index
        :type region_index: int
        
        :return: Returns a new list of voronoi regions with infinite vertices replaced.
        """
        try:
            region_point_index, = np.argwhere(self.vor.point_region == region_index)[0]
        except:
            raise ValueError("No region point index found in VoronoiTessesllation"
                             " for Adaptive Surrogate Generation. Try a different random seed.")
        
        region_vertices = []
        if -2 in region:
            finite_indices = [v for v in region if v >= 0]
            if len(finite_indices) > 0:
                finite_vertices = self.vor.vertices[finite_indices]
            new_vertices = self.snip_ridge_vertices(\
                region_index, region_point_index, region_tuple)

            # Replace the infinite vertex
            if len(finite_indices) > 0:
                region_vertices = np.concatenate((finite_vertices, new_vertices))
            elif len(new_vertices) > 0:
                region_vertices = new_vertices
            else:
                return None

        else:
            finite_indices = [idx for idx in region if idx >= 0]
            region_vertices = self.vor.vertices[finite_indices]
        return region_vertices

    def snip_ridge_vertices(self, region_index, region_point_index, region_tuple):
        """
        Replace out-of-bounds ridge vertices with boundary-hull intersections.

        Ridges containing SciPy's ``-1`` infinite vertex marker are skipped rather
        than accidentally indexing ``vertices[-1]``.
        """
        region_dict = {old_idx: new_idx for old_idx, new_idx in region_tuple}
        ridge_ids = self._ridge_ids_for_region_point(region_point_index)
        new_vertices = []

        for ridge_id in ridge_ids:
            new_vertices += self._snip_one_ridge(
                ridge_id,
                region_dict,
                region_index,
            )

        if len(new_vertices) == 0:
            return np.empty((0, self.ndim))

        return np.asarray(new_vertices, dtype=float)

    def _ridge_ids_for_region_point(self, region_point_index):
        """
        Return ridge indices associated with a Voronoi seed point.
        """
        return np.argwhere(self.vor.ridge_points == region_point_index)[:, 0]

    def _snip_one_ridge(self, ridge_id, region_dict, region_index):
        """
        Return clipped vertices generated from one Voronoi ridge.

        Ridges containing SciPy's ``-1`` infinite vertex marker are skipped to avoid
        accidental negative indexing.
        """
        ridge_vertices = self.vor.ridge_vertices[ridge_id]

        if -1 in ridge_vertices:
            return []

        updated_vertices = [region_dict.get(idx, idx) for idx in ridge_vertices]
        clipped_vertices = []

        for outside_pos, inside_pos in self._outside_inside_ridge_edges(updated_vertices):
            clipped = self._snip_one_edge(
                ridge_vertices[inside_pos],
                ridge_vertices[outside_pos],
                region_index,
            )

            if clipped is not None:
                clipped_vertices.append(clipped)

        return self._unique_vertex_list(clipped_vertices)
    
    def _outside_inside_ridge_edges(self, updated_vertices):
        """
        Yield ridge-edge positions with one outside and one finite inside vertex.

        The returned tuple is always ``(outside_position, inside_position)``.
        Both edge orientations are handled.
        """
        positions = list(range(len(updated_vertices)))

        for a, b in zip(positions, np.roll(positions, -1)):
            a_outside = updated_vertices[a] == -2
            b_outside = updated_vertices[b] == -2
            a_finite = updated_vertices[a] >= 0
            b_finite = updated_vertices[b] >= 0

            if a_outside and b_finite:
                yield a, b
            elif b_outside and a_finite:
                yield b, a

    def _snip_one_edge(self, origin_vertex_id, end_vertex_id, region_index):
        """
        Clip one bounded-to-outside Voronoi edge to the parameter-domain hull.
        """
        if origin_vertex_id < 0 or end_vertex_id < 0:
            return None

        origin = self.vor.vertices[origin_vertex_id]
        end = self.vor.vertices[end_vertex_id]
        direction = self.get_normal_ray_direction(origin, end)

        new_vertex = self.find_boundary_hull_ray_crossings(direction, origin)

        if not self._new_vertex_is_valid_for_region(new_vertex, region_index):
            return None

        return new_vertex

    def _new_vertex_is_valid_for_region(self, new_vertex, region_index):
        """
        Return True if a clipped vertex is finite and inside the boundary hull.

        The explicit Voronoi-region membership check is intentionally omitted here
        because clipped boundary intersections can be numerically assigned to an
        adjacent or ghost-influenced region even when they are valid for bounding
        the physical cell.
        """
        if new_vertex is None:
            return False

        new_vertex = np.asarray(new_vertex, dtype=float)

        if not np.isfinite(new_vertex).all():
            return False

        return self.bhullD.find_simplex(new_vertex, tol=1e-10) >= 0

    def _unique_vertex_list(self, vertices):
        """
        Return a list of unique vertices from a possibly duplicated vertex list.
        """
        if len(vertices) == 0:
            return []

        vertices = np.unique(np.asarray(vertices, dtype=float), axis=0)
        return [vertex for vertex in vertices]

    def get_normal_ray_direction(self, ray_origin, ray_end):
        """
        Return the normalized direction from ``ray_origin`` to ``ray_end``.
        """
        ray_direction = ray_end - ray_origin
        norm = np.linalg.norm(ray_direction)

        if norm <= 0:
            raise ValueError("Cannot normalize a zero-length Voronoi ray direction.")

        return ray_direction / norm
        
    def find_boundary_hull_ray_crossings(self, U, z):
        """Find where a ray crosses the convex hull of the boundary.

        :param U: Ray direction.
        :type U: np.ndarray
        
        :param z: Ray origin.
        :type z: np.ndarray

        :return: Returns a list of intersection points with the convex hull.
        """
        V = self.boundary_hull_V
        b = self.boundary_hull_b
        denom = np.dot(V, U)
        num = -(b + np.dot(V, z))
        alpha = num[denom!=0] / denom[denom!=0]
        if not np.any(alpha > 0):
           return None 
            
        return np.min(alpha[alpha >0]) * U + z

    def find_furthest_vertex(self, region_index, identify_outside_vertices=True):
        """
        Find the Voronoi vertex farthest from the region seed.
        """
        self.raise_if_invalid_region_index(region_index)

        vertices = self.get_region_vertices(
            region_index,
            identify_outside_vertices=identify_outside_vertices,
        )

        if vertices is None or len(vertices) == 0:
            return None, None

        centroid = self.get_region_seed(region_index)
        distances = np.linalg.norm(vertices - centroid, axis=1)
        furthest_vertex_index = int(np.argmax(distances))

        return vertices, furthest_vertex_index

    def raise_if_invalid_region_index(self, region_index):
        """
        Validate that a Voronoi region index refers to a valid, nonempty region.

        This method is used before accessing ``self.vor.regions[region_index]``.
        SciPy Voronoi objects may contain empty regions, and invalid region indices
        can otherwise produce confusing downstream errors when selecting adaptive
        sample locations.

        :param region_index: Index into ``self.vor.regions``.
        :type region_index: int or numpy.integer

        :raises TypeError: If ``region_index`` is not an integer.
        :raises ValueError: If ``region_index`` is outside the valid range or refers
            to an empty Voronoi region.
        """
        if not isinstance(region_index, (int, np.integer)):
            raise TypeError(
                "Voronoi region index must be an integer. "
                f"Received type '{type(region_index)}'."
            )

        n_regions = len(self.vor.regions)

        if region_index < 0 or region_index >= n_regions:
            raise ValueError(
                f"Invalid Voronoi region index {region_index}. "
                f"Valid region indices are in [0, {n_regions - 1}]."
            )

        if len(self.vor.regions[region_index]) == 0:
            raise ValueError(
                f"Voronoi region {region_index} is empty and cannot be used "
                "to select a new adaptive sample."
            )

    def get_region_seed(self, region_index):
        """Given a region_index, return the seed of the Voronoi tesselation that
        belongs to the region.

        :param region_index: Region index.
        :type region_index: int
        
        :return: Returns the Voronoi seed that belongs to the indexed region.
        """

        point_index, = np.where(self.vor.point_region == region_index)
        return np.atleast_2d(self.vor.points[point_index[0]])

    def get_voronoi_region(self, point_array):
        """Given an array of points, return the region of the Voronoi tesselation that the
        points belongs to. If a point lies on a ridge or vertice, multiple regions are 
        returned for that point.

        :param point: An array of points to find the region of.
        :type point: nd_array

        :return: Returns a list of lists, where each sublist contains the Voronoi 
            region(s) that contains the point. A point on a ridge has a sublist with 
            two regions (for 2D). A point on a vertice has a sublist
            with 3 regions (for 2D)
        """
        point_array = np.atleast_2d(point_array)
        region_index = []
        for point in point_array:
            point_already_exists = np.any(np.all(self.vor.points == point, axis=1))
            if point_already_exists:
                seed_index, = np.where(np.all(self.vor.points == point, axis=1))
            else:
                seed_index = self.get_closest_seed(point)
            
            # Get the region index for the point
            region_index.append(self.vor.point_region[seed_index].tolist())
        region_index = [sublist if sublist else [np.inf] for sublist in region_index]
        return region_index
    
    def get_closest_seed(self, point):
        """Return the index of the seed of the Voronoi cell that contains the given point.
            If the point lies on a ridge or vertex, multiple seeds are returned."""
        closest_seed_index = self.get_closest_point(self.vor.points, point)
        return closest_seed_index[0]

    def get_closest_point(self, candidates, target_point):
        """Return the index of the candidate point that has the min distance
           from the target point"""
        distances = np.linalg.norm(candidates - target_point, axis=1)
        min_dist = min(distances)
        closest_candidate_index = np.where(np.isclose(distances, min_dist, rtol=0, atol=1e-10))
        return closest_candidate_index        

    def remove_invalid_rows(self, arr):
        """ Remove points from NumPy array that contain NaN or infinite values."""
        if not isinstance(arr, np.ndarray):
            raise TypeError("Input to remove_invalid_rows must be a NumPy array.")
        arr = np.atleast_2d(arr)
        
        # Create a boolean mask for valid rows (no NaN, no inf, no -inf)
        mask = np.isfinite(arr).all(axis=1)
        return arr[mask]
    
    def add_points(self, points):
        """
        Add physical sample points and rebuild the Voronoi tessellation.

        The Voronoi tessellation contains two classes of seed points:

        * physical sample points stored in ``self.points``; and
        * auxiliary ghost points stored in ``self._ghost_points``.

        Only physical sample points should be added through this method. After new
        physical points are added, all derived tessellation state is rebuilt,
        including ghost points, the combined point array, the SciPy Voronoi object,
        ghost-point bookkeeping, and boundary-region bookkeeping.

        Rebuilding all derived state is necessary because SciPy Voronoi region
        indices and point-region mappings can change whenever points are added. If
        only ``self.vor`` is rebuilt while cached arrays such as ``self._boo`` or
        ``self.boundary_regions`` are left unchanged, later calls to region and
        vertex-selection methods can use stale indices and select incorrect adaptive
        samples.

        Invalid rows containing ``NaN`` or infinite values are discarded. Duplicate
        physical points are removed before rebuilding the tessellation.

        :param points: New physical sample point or points to add. A single point
            may be supplied with shape ``(n_dimensions,)``. Multiple points should
            have shape ``(n_points, n_dimensions)``.
        :type points: numpy.ndarray

        :raises TypeError: If ``points`` is not a NumPy array.
        :raises ValueError: If the supplied points do not have the same dimension as
            the existing physical sample points.

        :ivar points: Updated unique physical sample points.
        :vartype points: numpy.ndarray

        :ivar _ghost_points: Regenerated ghost points corresponding to the current
            physical sample set and parameter bounds.
        :vartype _ghost_points: numpy.ndarray

        :ivar _all_points: Combined physical and ghost seed points used to construct
            the Voronoi tessellation.
        :vartype _all_points: numpy.ndarray

        :ivar vor: Rebuilt SciPy Voronoi tessellation.
        :vartype vor: scipy.spatial.Voronoi

        :ivar _boo: Boolean list marking which entries of ``_all_points`` are ghost
            points.
        :vartype _boo: list[bool]

        :ivar boundary_regions: Updated Voronoi region indices associated with the
            domain boundary points.
        :vartype boundary_regions: list[list[int]]
        """
        points = self._prepare_points_to_add(points)

        if points.size == 0:
            logger.warning("No finite points were added to the Voronoi tessellation.")
            return

        self._append_unique_physical_points(points)
        self._rebuild_voronoi_state()

    def _prepare_points_to_add(self, points):
        """
        Validate and sanitize points before adding them to the tessellation.
        """
        points = np.asarray(points, dtype=float)

        if points.size == 0:
            return np.empty((0, self.ndim))

        points = self.remove_invalid_rows(np.atleast_2d(points))
        self._check_added_point_dimension(points)

        return points

    def _check_added_point_dimension(self, points):
        """
        Validate that new points have the tessellation dimension.
        """
        if points.size > 0 and points.shape[1] != self.points.shape[1]:
            raise ValueError("New points have the wrong dimension.")

    def _append_unique_physical_points(self, points):
        """
        Append physical points and remove duplicates.
        """
        self.points = np.unique(np.vstack((self.points, points)), axis=0)

    def _rebuild_voronoi_state(self):
        """
        Rebuild ghost points, combined points, Voronoi object, and cached mappings.
        """
        from scipy.spatial import Voronoi

        self.create_ghost_points()
        self._all_points = np.vstack([self.points, self._ghost_points])
        self.vor = Voronoi(self._all_points, incremental=self.incremental)
        self.ghost_busters()
        self.boundary_regions = self.get_voronoi_region(self.boundary_points)


class KFoldCrossValidation:
    def __init__(
        self,
        nsplits,
        group_kfold,
        interpolation_field,
        interpolation_values,
        scale,
        metric,
        target_field,
        param_names,
        surrogate_options,
        random_seed=None,
    ):
        """
        Initialize the K-fold cross-validation helper.
        """
        self.nsplits = nsplits
        self.group_kfold = group_kfold
        self.scale = scale
        self.metric = metric
        self.interpolation_field = interpolation_field
        self.interpolation_values = interpolation_values
        self.target_field = target_field
        self.param_names = param_names
        self.surrogate_options = surrogate_options
        self.random_seed = random_seed

    def _check_nsplits(self, training_params):
        """
        Normalize the K-fold split count for the current training sample count.
        """
        self.nsplits = _get_valid_kfold_split_count(
            self.nsplits,
            training_params.shape[0],
        )
            
    def perform_kfold_cv(self, training_params, training_data, groups):
        """
        Perform K-fold or grouped K-fold cross validation.
        """
        self._check_nsplits(training_params)
        splits = self._make_cv_splits(training_params, training_data, groups)

        return self._evaluate_cv_splits(training_params, training_data, splits)


    def _make_cv_splits(self, training_params, training_data, groups):
        """
        Create the K-fold or GroupKFold split generator.
        """
        if self.group_kfold:
            return self._make_group_kfold_splits(training_params, training_data, groups)

        return self._make_standard_kfold_splits(training_params)


    def _make_group_kfold_splits(self, training_params, training_data, groups):
        """
        Create grouped K-fold splits.
        """
        from sklearn.model_selection import GroupKFold

        if groups is None:
            raise RuntimeError("GroupKFold requested but no groups were provided.")

        cv = GroupKFold(n_splits=self.nsplits)
        return cv.split(training_params, training_data, groups)


    def _make_standard_kfold_splits(self, training_params):
        """
        Create reproducible shuffled K-fold splits.
        """
        from sklearn.model_selection import KFold

        cv = KFold(
            n_splits=self.nsplits,
            shuffle=True,
            random_state=self.random_seed,
        )
        return cv.split(training_params)


    def _evaluate_cv_splits(self, training_params, training_data, splits):
        """
        Evaluate all K-fold splits and return results in a dictionary.
        """
        from joblib import Parallel, delayed

        results = Parallel(n_jobs=1)(
            delayed(self.evaluate_fold)(tr, te, training_params, training_data, i)
            for i, (tr, te) in enumerate(splits)
        )

        return {i: result for i, result in enumerate(results)}

    def evaluate_fold(self, train_index, test_index, X, y, kfold_count):
        """
        Evaluate one K-fold split using physical response-space error.

        The previous implementation ranked folds using latent surrogate scores from
        ``fold_surrogate._latent_scores['test']``. The KFCV-Voronoi method described
        in the paper ranks folds by the physical prediction error on the held-out
        samples,

        .. math::

            e_i^{KF}
            =
            \\sum_{s_j \\in kf_i}
            \\left|
            y(s_j) - \\hat{y}_{S \\setminus kf_i}(s_j)
            \\right|.

        This implementation evaluates the fold surrogate at the held-out physical
        parameter samples and compares those predictions directly with the held-out
        physical responses.

        :param train_index: Indices used to train the fold surrogate.
        :type train_index: array-like

        :param test_index: Held-out fold indices.
        :type test_index: array-like

        :param X: Full parameter sample matrix.
        :type X: numpy.ndarray

        :param y: Full list of physical model-evaluation dictionaries.
        :type y: list[dict]

        :param kfold_count: Fold counter used for logging.
        :type kfold_count: int

        :return: Tuple containing the physical fold error and the held-out indices.
        :rtype: tuple[float, numpy.ndarray]
        """
        logger.info(
            f"\tEvaluating physical '{self.metric}' error for surrogate "
            f"for kfold cross validation set {kfold_count}..."
        )

        info = self.extract_fold_info(train_index, test_index, X, y)
        train_eval_info, test_eval_info, X_test, y_test = info

        fold_surrogate = _fit_surrogate_model(
            train_eval_info,
            self.interpolation_field,
            self.interpolation_values,
            test_eval_info,
            self.target_field,
            "kfold_validation_surrogate.joblib",
            logger_on=False,
            **self.surrogate_options,
        )

        error = _calculate_physical_cv_error(
            fold_surrogate,
            X_test,
            y_test,
            self.target_field,
            self.interpolation_field,
            self.interpolation_values,
            self.metric,
            self.scale,
        )

        logger.info(f"\t\tphysical error = {error}")
        return error, test_index
   
    def extract_fold_info(self, train_index, test_index, X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train = [y[i] for i in train_index]
        y_test = [y[i] for i in test_index]
        train_res, test_res = _setup_studies_for_cv(self.param_names,
                                                    X_train, X_test, y_train, y_test)
        return train_res, test_res, X_test, y_test


class LeaveOneOutCrossValidation:
    def __init__(self, scale, metric, interpolation_field, 
                 interpolation_values, target_field, par_names, surrogate_options):
        """
        Initialize the LOOCV.
        """
        self.scale = scale
        self.metric = metric
        self.interpolation_field = interpolation_field
        self.interpolation_values = interpolation_values
        self.target_field = target_field
        self.par_names = par_names
        self.surrogate_options = surrogate_options

    def perform_loocv(self, X, y, indices):

        """Perform Leave-One-Out Cross-Validation.

        :param X: Feature matrix (training samples).
        :type X: np.ndarray
        
        :param y: Target values (ground truth).
        :type y: np.ndarray

        :return: Returns the index of the sample with the greatest 
            prediction error and the corresponding error value.
            tuple: (index_of_max_error, max_error)
        """
        from joblib import Parallel, delayed
        
        loo_results = Parallel(n_jobs=1)(
            delayed(self.evaluate_loo_sample)(X, y, i)
            for i in indices
        )

        loo = {loo_idx: result for loo_idx, result in enumerate(loo_results)}
        return loo
    
    def evaluate_loo_sample(self, X, y, index):
        """
        Evaluate one leave-one-out split using physical response-space error.

        The previous implementation ranked samples using latent surrogate scores.
        The CV-Voronoi and KFCV-Voronoi methods require the physical prediction
        error at the omitted sample,

        .. math::

            e_i^{LOO}
            =
            \\left|
            y(s_i) - \\hat{y}_{S \\setminus s_i}(s_i)
            \\right|.

        This implementation builds the leave-one-out surrogate, evaluates it at the
        omitted physical parameter sample, and compares the prediction directly with
        the omitted physical model response.

        :param X: Full parameter sample matrix.
        :type X: numpy.ndarray

        :param y: Full list of physical model-evaluation dictionaries.
        :type y: list[dict]

        :param index: Index of the omitted sample.
        :type index: int

        :return: Tuple containing the physical LOO error and omitted sample index.
        :rtype: tuple[float, int]
        """
        logger.info(
            f"\tEvaluating physical '{self.metric}' error for surrogate "
            f"leaving out sample {index}"
        )

        info = self.extract_loo_info(index, X, y)
        train_eval_info, test_eval_info, X_test, y_test = info

        fold_surrogate = _fit_surrogate_model(
            train_eval_info,
            self.interpolation_field,
            self.interpolation_values,
            test_eval_info,
            self.target_field,
            "kfold_validation_surrogate.joblib",
            logger_on=False,
            **self.surrogate_options,
        )

        error = _calculate_physical_cv_error(
            fold_surrogate,
            X_test,
            y_test,
            self.target_field,
            self.interpolation_field,
            self.interpolation_values,
            self.metric,
            self.scale,
        )

        logger.info(f"\t\tphysical error = {error}")
        return error, index

    def extract_loo_info(self, index, X, y):
        X_train = np.delete(X, index, axis=0)
        y_train = y.copy()
        del y_train[index]
        X_test = X[index].reshape(1, -1)  # Reshape for a single sample
        y_test = [y[index]]
        train_res, test_res = _setup_studies_for_cv(self.par_names,
                                                    X_train, X_test,
                                                    y_train, y_test)
        return train_res, test_res, X_test, y_test


def _setup_studies_for_cv(p_names, train_samples, test_samples,
                               train_evals, test_evals):
    res = _get_parameter_and_simulation_hist(p_names, train_samples, train_evals)
    test_res = _get_parameter_and_simulation_hist(p_names, test_samples, test_evals)
    return res, test_res


def _get_parameter_and_simulation_hist(p_names, p_samples, m_evals):
    from matcal.core.study_base import StudyResults
    p_hist = _format_parameter_hist(p_names, p_samples)
    res_hist = _format_parameter_evaluations(m_evals)
    res = StudyResults()
    res._update_parameter_history(p_hist, list(p_hist.keys()))
    res._update_simulation_history(res_hist, 'cv')
    return res


def _format_parameter_hist(names, p_samples):
    n_samples = p_samples.shape[0]
    params = OrderedDict()
    for idx in range(n_samples):
        params[f"eval_{idx}"] = OrderedDict()
        for n_idx, param_name in enumerate(names):
            params[f"eval_{idx}"][param_name] = p_samples[idx, n_idx]
    return params 


def _format_parameter_evaluations(model_evals):
    from matcal.core.data import convert_dictionary_to_data, DataCollection
    results_hist = DataCollection("CV data collection")
    for eval in model_evals:
        results_hist.add(convert_dictionary_to_data(eval))
    return results_hist


def _get_surrogate_metric(latent_scores_test, metric):
    combined_score = []
    for field_idx, field_name in enumerate(latent_scores_test):
        if isinstance(latent_scores_test[field_name], (dict, OrderedDict)):
            combined_score += list(latent_scores_test[field_name][metric])
    if combined_score == len(combined_score)*[None]:
        return np.nan
    elif metric == 'nlpd':
        return np.sum(combined_score)
    else:
        return np.mean(combined_score)

def _extract_physical_response_matrix(model_evals, target_field,
                                      interpolation_field=None,
                                      interpolation_values=None):
    """
    Extract physical target-response values from a list of model-evaluation
    dictionaries.

    The cross-validation adaptive-sampling criteria should be based on physical
    response error,

    .. math::

        y(s_i) - \\hat{y}_{S \\setminus s_i}(s_i),

    not on latent-space surrogate diagnostics. This helper converts the held-out
    model responses into a dense array suitable for direct comparison with
    surrogate predictions.

    If ``interpolation_values`` are provided and the stored target response is
    defined on a different independent-variable grid, the target response is
    interpolated onto ``interpolation_values`` using ``interpolation_field``.

    :param model_evals: List of model-evaluation dictionaries. Each dictionary
        must contain ``target_field`` and, when interpolation is needed,
        ``interpolation_field``.
    :type model_evals: list[dict]

    :param target_field: Name of the physical response field to compare.
    :type target_field: str

    :param interpolation_field: Name of the independent-variable field used for
        interpolation, e.g. ``"time"`` or ``"x"``.
    :type interpolation_field: str or None

    :param interpolation_values: Independent-variable values at which the
        surrogate response is evaluated.
    :type interpolation_values: array-like or None

    :return: Physical response array with shape ``(n_samples, n_qois)``.
    :rtype: numpy.ndarray
    """
    responses = []

    if interpolation_values is not None:
        interpolation_values = np.asarray(interpolation_values, dtype=float).reshape(-1)

    for eval_data in model_evals:
        if target_field not in eval_data:
            raise KeyError(
                f"Target field '{target_field}' was not found in a "
                "cross-validation model evaluation."
            )

        target_response = np.asarray(eval_data[target_field], dtype=float).reshape(-1)

        needs_interpolation = (
            interpolation_values is not None
            and target_response.size != interpolation_values.size
        )

        if needs_interpolation:
            if interpolation_field is None:
                raise RuntimeError(
                    "Cannot interpolate held-out physical response because "
                    "interpolation_field is None."
                )

            if interpolation_field not in eval_data:
                raise KeyError(
                    f"Interpolation field '{interpolation_field}' was not found "
                    "in a cross-validation model evaluation."
                )

            source_x = np.asarray(eval_data[interpolation_field], dtype=float).reshape(-1)

            if source_x.size != target_response.size:
                raise RuntimeError(
                    "Cannot interpolate held-out physical response because the "
                    f"independent field '{interpolation_field}' has length "
                    f"{source_x.size}, but target field '{target_field}' has "
                    f"length {target_response.size}."
                )

            sort_idx = np.argsort(source_x)
            source_x = source_x[sort_idx]
            target_response = target_response[sort_idx]

            target_response = np.interp(
                interpolation_values,
                source_x,
                target_response,
            )

        responses.append(target_response)

    return np.asarray(responses, dtype=float)


def _evaluate_surrogate_physical_response(surrogate, X_test, target_field):
    """
    Evaluate a surrogate at held-out parameter samples and return its physical
    target-field response.

    The returned array is normalized to shape ``(n_samples, n_qois)`` so that it
    can be directly compared with the held-out physical model responses.

    :param surrogate: Surrogate object returned by ``_fit_surrogate_model``.
    :type surrogate: object

    :param X_test: Held-out parameter samples with shape
        ``(n_test_samples, n_parameters)``.
    :type X_test: numpy.ndarray

    :param target_field: Name of the physical target field.
    :type target_field: str

    :return: Surrogate predictions with shape ``(n_test_samples, n_qois)``.
    :rtype: numpy.ndarray
    """
    X_test = np.asarray(X_test, dtype=float)
    n_test_samples = X_test.shape[0]

    surrogate_results = surrogate(X_test, batch_evaluate=True)

    if target_field not in surrogate_results:
        raise KeyError(
            f"Target field '{target_field}' was not returned by the surrogate."
        )

    predicted_response = np.asarray(surrogate_results[target_field], dtype=float)

    if predicted_response.ndim == 0:
        predicted_response = predicted_response.reshape(1, 1)

    elif predicted_response.ndim == 1:
        if n_test_samples == 1:
            predicted_response = predicted_response.reshape(1, -1)
        else:
            predicted_response = predicted_response.reshape(n_test_samples, -1)

    elif predicted_response.ndim == 2:
        if predicted_response.shape[0] == n_test_samples:
            pass
        elif predicted_response.shape[1] == n_test_samples:
            predicted_response = predicted_response.T
        else:
            raise RuntimeError(
                "Could not orient surrogate prediction array for physical "
                f"cross-validation error. Prediction shape is "
                f"{predicted_response.shape}; expected one dimension to equal "
                f"the number of held-out samples, {n_test_samples}."
            )

    else:
        raise RuntimeError(
            "Physical cross-validation error only supports scalar, vector, or "
            f"matrix surrogate responses. Received response with shape "
            f"{predicted_response.shape}."
        )

    return predicted_response


def _apply_physical_response_scaling(true_response, predicted_response, scale):
    """
    Apply optional scaling to physical responses before error calculation.

    ``scale`` is intended to normalize response magnitudes before computing
    cross-validation errors. If ``scale`` is ``None`` or ``1``, responses are
    returned unchanged.

    The legacy string option ``"cbrt"`` is supported for compatibility.

    :param true_response: Held-out physical response.
    :type true_response: numpy.ndarray

    :param predicted_response: Surrogate-predicted physical response.
    :type predicted_response: numpy.ndarray

    :param scale: Response scaling option.
    :type scale: float, str, numpy.ndarray, or None

    :return: Scaled true and predicted responses.
    :rtype: tuple[numpy.ndarray, numpy.ndarray]
    """
    if scale is None:
        return true_response, predicted_response

    if isinstance(scale, str):
        scale_lower = scale.lower().strip()
        if scale_lower == "cbrt":
            return np.cbrt(true_response), np.cbrt(predicted_response)
        raise ValueError(
            f"Unsupported physical response scale option '{scale}'."
        )

    scale = np.asarray(scale, dtype=float)

    if np.any(scale <= 0):
        raise ValueError("Physical response scale values must be positive.")

    return true_response / scale, predicted_response / scale


def _calculate_physical_cv_error(surrogate, X_test, y_test, target_field,
                                 interpolation_field, interpolation_values,
                                 metric="rmse", scale=1.0):
    """
    Calculate cross-validation error in physical response space.

    This replaces the previous latent-score-based adaptive-sampling criterion.
    The returned error is based on the held-out physical response and the
    surrogate prediction at the same held-out parameter locations.

    For K-fold CV, ``X_test`` and ``y_test`` contain all samples in the held-out
    fold. For LOOCV, they contain one held-out sample.

    Supported metrics are:

    * ``"rmse"``: root mean squared physical response error.
    * ``"mae"`` or ``"abs"``: mean absolute physical response error.
    * ``"sum_abs"``: sum of absolute physical response errors, closest to the
      paper's stated error form.
    * ``"nrmse"``: normalized root mean squared physical response error.
    * ``"nlpd"``: accepted for backward compatibility, but evaluated as physical
      RMSE because true NLPD requires predictive variances.

    :param surrogate: Surrogate object returned by ``_fit_surrogate_model``.
    :type surrogate: object

    :param X_test: Held-out parameter samples.
    :type X_test: numpy.ndarray

    :param y_test: Held-out model-evaluation dictionaries.
    :type y_test: list[dict]

    :param target_field: Name of the physical response field.
    :type target_field: str

    :param interpolation_field: Independent-variable field used to align
        physical responses with surrogate outputs.
    :type interpolation_field: str

    :param interpolation_values: Independent-variable values at which the
        surrogate response is evaluated.
    :type interpolation_values: array-like

    :param metric: Physical error metric.
    :type metric: str

    :param scale: Optional physical response scale.
    :type scale: float, str, numpy.ndarray, or None

    :return: Scalar physical cross-validation error.
    :rtype: float
    """
    true_response = _extract_physical_response_matrix(
        y_test,
        target_field,
        interpolation_field,
        interpolation_values,
    )

    predicted_response = _evaluate_surrogate_physical_response(
        surrogate,
        X_test,
        target_field,
    )

    if true_response.shape != predicted_response.shape:
        raise RuntimeError(
            "Held-out physical response and surrogate prediction have "
            "different shapes during cross-validation error calculation. "
            f"True response shape: {true_response.shape}. "
            f"Predicted response shape: {predicted_response.shape}."
        )

    true_response, predicted_response = _apply_physical_response_scaling(
        true_response,
        predicted_response,
        scale,
    )

    residual = true_response - predicted_response

    metric = metric.lower().strip()

    if metric == "rmse":
        return float(np.sqrt(np.mean(residual ** 2)))

    if metric == "nlpd":
        # Backward-compatible behavior. True physical NLPD would require a
        # physical predictive variance. For adaptive region ranking, use physical
        # RMSE rather than latent-space NLPD.
        return float(np.sqrt(np.mean(residual ** 2)))

    if metric in ("mae", "abs"):
        return float(np.mean(np.abs(residual)))

    if metric == "sum_abs":
        return float(np.sum(np.abs(residual)))

    if metric == "nrmse":
        denom = np.sum(true_response ** 2)
        if denom <= 0:
            return float(np.sqrt(np.mean(residual ** 2)))
        return float(np.sqrt(np.sum(residual ** 2) / denom))

    raise ValueError(
        "Unsupported physical cross-validation metric "
        f"'{metric}'. Supported metrics are 'rmse', 'mae', 'abs', "
        "'sum_abs', 'nrmse', and backward-compatible 'nlpd'."
    )
