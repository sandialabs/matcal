"""
This module contains adaptive surrogates. 
"""
from collections import OrderedDict
import copy
import numpy as np
import os
from sklearn.metrics import r2_score

from matcal.core.logger import initialize_matcal_logger
from matcal.core.objective import SimulationResultsSynchronizer
from matcal.core.parameter_studies import HaltonStudy
from matcal.core.qoi_extractor import UserDefinedExtractor
from matcal.core.state import State
from matcal.core.study_base import StudyResults
from matcal.core.utilities import (check_value_is_positive_integer, 
                                   check_value_is_positive_integer_or_none,
                                   check_value_is_array_like_of_reals, 
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
                                    _convert_param_array_to_dict)

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

    By default, only the best surrogate, as measured by RMSE on the test set,
    is retained.
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

        By default, only the best surrogate by RMSE is retained.

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

        Retain only the best surrogate by RMSE:

        >>> study.set_surrogate_storage_options(best_n_surrogates=1)

        Retain the best five surrogates by maximum absolute error:

        >>> study.set_surrogate_storage_options(
        ...     best_n_surrogates=5,
        ...     score_metric="max_error",
        ... )

        Retain every tenth adaptive batch surrogate:

        >>> study.set_surrogate_storage_options(
        ...     best_n_surrogates=None,
        ...     save_every_n_batches=10,
        ... )

        Retain the best two surrogates and every fifth batch surrogate:

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

        if self._test_responses.shape[1] > 1:
            score = r2_score(self._test_responses, surrogate_values)
        else:
            score = np.nan

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
                logger.info(f"Root mean squared error converged! "+
                            f"\nFinal RMSE: {self._surrogate.rmse_history[-1]}")
                stop=True
            elif np.abs(self._surrogate.max_error_history[-1]) <=self._max_abs_error_goal:
                logger.info(f"Max absolute error score converged! "+
                            f"\nFinal max error: {self._surrogate.max_error_history[-1]}")
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
        if ".joblib" not in filename:
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

        By default, only the best surrogate by RMSE is retained.

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
        """Initialize the VoronoiAdaptiveSurrogateStudy
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

        self._current_surrogate_score = {}
        self._current_surrogate_score['score'] = []
        self._current_surrogate_score['nlpd'] = []
        self._current_surrogate_score['rmse'] = []
        self._max_fold_error_indices = None
        self._surrogate_options = {}
        self._seed = None
            
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

    def set_voronoi_sampling_options(self, voronoi_type='full', 
                                     finite_only=False, 
                                     iterative_updates = True, 
                                     thin=None,
                                     random_selection=None):
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
        check_value_is_nonempty_str(voronoi_type, "voronoi_type")
        if voronoi_type.lower() not in ['full', 'local']:
            raise ValueError(f"Voronoi type must be either 'full' or 'local', recieved '{voronoi_type}'.")
        else:
            self._voronoi_type = voronoi_type.lower()
        check_value_is_bool(finite_only, "finite_only")
        self._finite_only = finite_only
        check_value_is_bool(iterative_updates, "iterative_updates")
        self._iterative_updates = iterative_updates
        if thin is not None:
            check_value_is_positive_integer(thin, "thin")
            self._thin = thin
        if random_selection is not None:
            check_value_is_positive_integer(random_selection, "random_selection")
            self._random_selection = random_selection
        if self._random_selection is not None and self._thin is not None:
            raise ValueError("Only one of 'thin' and 'random_selection' can be activated. Not both.")
    
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

    def set_cross_validation_options(self, nsplits=5, nmax_folds=3, nmax_loo=10, cv_scale=1.0,
                                     cv_metric='rmse', group_kfold=False):
        """Set options for cross validation. Properties that can be altered are listed below.
        
        :param nsplits: The number of folds to use in k-fold cross validation. 
            If nsplits = 0, k-fold cross-validation is skipped entirely and new samples
            are instead selected from every region of the Voronoi tessellation defined 
            by the current set of training samples.
        :type nsplits: int
        
        :param nmax_folds: Points in the folds with the highest k-fold error 
            (the top nmax_folds) define the Voronoi regions from which new samples 
            will be drawn. 
        :type nmax_folds: int
        
        :param nmax_loo: Points with the largest leave-one-out cross-validation (LOOCV)
            errors (the top nmax_loo). These define the Voronoi regions from which new 
            samples will be drawn. If nmax_loo = 'all', then new samples are drawn from
            all Voronoi regions defined by nmax_folds, and leave-one-out cross-validation
            is not performed.
        :type nmax_loo: int or 'all'
        
        :param cv_scale: Optional scaling applied to output before calculating errors in
            cross-validation and leave-one-out cross-validation. This can be used to 
            balance error magnitude across dimensions or outputs.
        :type scale: float
        
        :param cv_metric: Determines which metric is used when computing errors during
            cross-validation. Supported options are:
                * rmse -- root mean squared error (Default)
                * nlpd -- negative log posterior density
        :type cv_metric: str
        
        :param group_kfold: If True, samples are grouped using k-means clustering
            prior to k-fold cross-validation so that nearby points are allways assigned
            to the same fold. This prevents spatially correllated points from being split
            across training and validation sets. If False, folds are assigned randomly
            by the standard KFold algorithm. 
        :type group_kfold: bool
        """
        check_value_is_nonnegative_integer(nsplits, "nsplits")
        self._nsplits = nsplits
        check_value_is_positive_integer(nmax_folds, "nmax_folds")
        self._nmax_folds = nmax_folds
        if isinstance(nmax_loo, str):
            if nmax_loo != 'all':
                raise ValueError(f"If the {__class__} 'nmax_loo' parameter is a string, "+
                    "it must be 'all'.")
        else:
            try:   
                check_value_is_positive_integer(nmax_loo, "nmax_loo")
            except TypeError:
                raise TypeError(f"The {__class__} 'nmax_loo' parameter must be a positive integer "+
                                f"or the string 'all'. Recieved value {nmax_loo}.")
            except  ValueError:
                raise ValueError(f"The {__class__} 'nmax_loo' parameter must be a positive integer "+
                                f"recieved value {nmax_loo}.")
        self._nmax_loo = nmax_loo
        check_value_is_positive_real(cv_scale, "cv_scale")
        self._cv_scale = cv_scale
        check_value_is_nonempty_str(cv_metric, "cv_metric")
        self._cv_metric = cv_metric
        valid_cv_metrics = ['rmse', 'nlpd']
        if self._cv_metric not in valid_cv_metrics:
            raise ValueError("cv_metric not implemented. 'cv_metric' must one of"
                             " 'rmse', 'nlpd'")
        check_value_is_bool(group_kfold, "group_kfold")
        self._group_kfold = group_kfold
    
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
        if self._surrogate_save_filename is None:
            self.set_surrogate_save_filename(f"{self._get_model_names()[0]}_voronoi_adaptive_surrogate.joblib") 
        self._build_boundary_hull()
        self.param_names = self._parameter_collection.get_item_names()
        training_params, training_data = self._run_initial_training_samples()
        batch_number = 0
        while not self._stopping_criterion_met(batch_number):
            logger.info(f"Active learning batch {batch_number+1}."
                        f"\nCurrently the surrogate is trained on "+
                        f"{self._nbatch_samples[-1]} samples.")
            logger.info("................................................................")
            new_points = self._create_voronoi_tess_and_choose_new_samples(batch_number, 
                                                                          training_params, 
                                                                          training_data)
            self._populate_parameter_evaluations(new_points)
            self._matcal_evaluate_parameter_sets_batch(self._parameter_sets_to_evaluate)
            training_params, training_data = self._train_surrogate_with_current_results()
            batch_number += 1
        return self._results

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

    def _create_voronoi_tess_and_choose_new_samples(self, iteration, training_params, 
                                                    training_data):
        """Perform Voronoi batch sampling based on the specified algorithm.
        
        :return: Returns a list of the new samples selected.
        """
        if self._nsplits > 0:
            # Step 1: Randomly sort existing samples into K-folds and perform KFold Cross Validation
            self._perform_kfold_cross_validation(training_params, training_data)

            # Step 2: Select the fold(s) with the n largest K-fold CV error(s)
            self._find_kfold_max_errors()
            
            if self._nmax_loo == 'all':
                worst_sample_locations = training_params[self._max_fold_error_indices]
            else:
                # Step 3: Use LOOCV to evaluate each sample within the selected fold(s)
                self._perform_loo_cross_validation(training_params, training_data)
                # Step 4: Identify the n sample(s) with the highest LOOCV error(s)
                worst_sample_locations = self._find_loo_max_errors(training_params)
                
        else:
            # Do not perform kfold or loo CV. New samples drawn for all Voroni regions.
            worst_sample_locations = training_params

        if self._thin is not None:
            # thin the new samples locations according to "thin"
            worst_sample_locations = worst_sample_locations[::self._thin, ...]
        elif self._random_selection is not None:
            # randomly select the new sample locations from the candidates in worst_sample_locations
            draw_n = np.min((int(0.5 * worst_sample_locations.shape[0]), self._random_selection))
            random_rows = np.random.choice(worst_sample_locations.shape[0], 
                                           size=draw_n, replace=False)
            worst_sample_locations = worst_sample_locations[random_rows, ...]

        self._worst_sample_locations = worst_sample_locations
        logger.info(f"Initializing voronoi/tree for batch {iteration}")
        self._create_voronoi_tess(training_params)
        return self._find_new_sample_locations()
        
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
        new_points = []
        logger.info("Finding new sample locations")
        for loc_idx, location in enumerate(self._worst_sample_locations):
            if np.mod(loc_idx, 100) == 0:
                logger.info(f"Drawing new sample from region index {loc_idx}"
                            f" of {len(self._worst_sample_locations)}.")

            if self._voronoi_type == 'full':
                # Identify corresponding voronoi cell
                region_index = self._voronoi_tessellation.get_voronoi_region(location)[0][0]

                # Step 5: Select the point within this sample’s Voronoi cell that
                # is furthest from existing samples
                region_vertices, furthest_vertex_index = \
                    self._voronoi_tessellation.find_furthest_vertex(region_index)
                if region_vertices is None:
                    continue
                furthest_vertex = region_vertices[furthest_vertex_index]

                # Step 6: Add the new point and update Voronoi tessellation
                if self._iterative_updates:
                    self._voronoi_tessellation.add_points(furthest_vertex)

                # Step 7: Update X
                new_points.append(furthest_vertex)

            elif self._voronoi_type == 'local':
                nneighbors = int(self._all_tree_points.shape[0] * 0.25)
                nearest_neighbors = self._tree.query(location, k=nneighbors)
                nn_points = self._all_tree_points[nearest_neighbors[1].squeeze()]
                nn_vor = VoronoiTessellation(nn_points, self._bounds, self._finite_only)
                nn_vor.build()
                nn_region = nn_vor.get_voronoi_region(location)[0][0]
                try:
                    nn_vert, nn_fvi = nn_vor.find_furthest_vertex(nn_region)
                except:
                    continue

                if nn_vert is None:
                    continue
                furthest_vertex = nn_vert[nn_fvi]
                new_points.append(furthest_vertex)
                if self._iterative_updates:
                    from scipy.spatial import KDTree
                    self._all_tree_points = np.vstack((self._all_tree_points, furthest_vertex))
                    self._tree = KDTree(self._all_tree_points)

        new_points = np.asarray(new_points)
        # Make sure all new points are unique
        unique_points = set(tuple(row) for row in new_points)
        new_points = np.asarray([list(row) for row in unique_points])
        new_points = self._check_points_within_bounds(new_points)
        return new_points

    def _check_points_within_bounds(self, points):
        # verify that all samples are within bounds
        lb = self._bounds[:,0]
        ub = self._bounds[:,1]
        mask = ((points >= lb) & (points <= ub)).all(axis=1)
        return points[mask]
        
    def _perform_kfold_cross_validation(self, training_params, training_data):
        self._kf = None
        logger.info("Performing kfold cross-validation")
        kfcv = KFoldCrossValidation(self._nsplits, self._group_kfold, self._independent_variable, 
                                    self._independent_variable_values, 
                                    self._cv_scale, self._cv_metric, self._target_field_name, 
                                    self.param_names, self._surrogate_options)
        groups = None
        if self._group_kfold:
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=self._nsplits, random_state=42)
            groups = kmeans.fit_predict(training_params)
        self._kf = kfcv.perform_kfold_cv(training_params, training_data, groups)

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
            self._worst_sample_locations = None
            max_loo_indices = self._find_indices_of_n_largest_errors()
            logger.info(f"\n\tWorst errors when the following sample indices " +
                        "are left out of training:\n"+
                        f"\t{max_loo_indices}\n")
            return training_params[max_loo_indices]
        
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

    def _find_indices_of_n_largest_errors(self):
        """Find the indices of the n largest values in an array of errors.

        :return: Returns an array of indices corresponding to the n largest errors.
        """
        nkeep = int(self._nmax_loo)
        # Create a list of (key, error, sample_index) tuples
        items = [(key, value[0], value[1]) for key, value in self._loo_errors.items()]
        # Sort the items based on the error in descending order
        sorted_items = sorted(items, key=lambda x: x[1], reverse=True)
        # Get the top n items
        top_n_items = sorted_items[:nkeep]
        # Extract the indices associated with the top n largest floats
        indices = [item[2] for item in top_n_items]
        return np.array(indices)

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
        """Reflect points nearest to the boundary hull across the nearest
        face of the boundary hull """

        boundary_points_stretched = self.boundary_points * stretchCoef
        self._ghost_points = boundary_points_stretched

        boundary_centroid = np.mean(self.boundary_points, axis=0)
        max_dist = np.max(np.linalg.norm(self.boundary_points - boundary_centroid, axis=1))
        self._ghost_points = \
            np.vstack([self._ghost_points, \
                boundary_centroid + centCoef * max_dist * np.eye(self.points.shape[1])])
        self._ghost_points =\
            np.vstack([self._ghost_points, \
                boundary_centroid - centCoef * max_dist * np.eye(self.points.shape[1])])

    def ghost_busters(self):
        """ Identify which points in self._all_points are ghost points"""
        self._boo = []
        for point in self._all_points:
            if point in self._ghost_points:
                self._boo.append(True)
            else:
                self._boo.append(False)

    def get_region_vertices(self, region_index, identify_outside_vertices=True):
        """Return the vertices of the Voronoi region."""
        region = self.vor.regions[region_index].copy()
        if -1 in region:
            logger.warning(f"Infinite vertice in Region {region_index}")
        
        if identify_outside_vertices:
            updated_region = self.identify_vertices_outside_bounds(region)
            if not -2 in updated_region and len(updated_region) > 0:
                region_vertices = self.vor.vertices[region]
            elif -2 in updated_region:
                if self.finite_only:
                    if max(updated_region) < 0:
                        region_vertices = None
                    else: 
                        region_vertices = \
                            np.asarray([self.vor.vertices[i]\
                                for i in updated_region if i > 0])
                else:
                    region_tuple_list = list(zip(region, updated_region))
                    region_vertices = \
                        self.replace_unbounded_vertices(updated_region, region_index, region_tuple_list)
            if region_vertices is not None:
                if not self.finite_only:
                    boundary_in_region = \
                        [i for i in np.arange(len(self.boundary_regions))\
                            if self.boundary_regions[i][0] == region_index]
                    if boundary_in_region:
                        boundary_vertices = self.boundary_points[boundary_in_region] 
                        region_vertices = np.concatenate((region_vertices, boundary_vertices))
                        unique_vertices = set(tuple(row) for row in region_vertices)
                        return np.asarray([list(row) for row in unique_vertices])
            return region_vertices

        elif not identify_outside_vertices:
            return self.vor.vertices[region]

    def get_voronoi_vertices(self, identify_outside_vertices=True):
        """Return the vertices of the Voronoi tessellation."""
        vertices = []
        for i, region in enumerate(self.vor.regions):
            try:
                region_point_index, = np.where(self.vor.point_region == i)[0]
            except:
                # empty region: Voronoi region for a point at infinity that was added internally
                continue
            if self._boo[region_point_index]:
                # region belongs to a ghost point
                continue
            elif -1 in region:
                logger.warning(f"Infinite vertice in Region {i}")

            if identify_outside_vertices:
                updated_region = self.identify_vertices_outside_bounds(region)
                if not -2 in updated_region and len(updated_region) > 0:
                    vertices.append(self.vor.vertices[region])
                elif -2 in updated_region:
                    if self.finite_only:
                        verts = np.asarray([self.vor.vertices[i] for i in updated_region if i > 0])
                        vertices.append(verts)
                    else:
                        region_tuple_list = list(zip(region, updated_region))
                        vertices.append(self.replace_unbounded_vertices(updated_region, i, region_tuple_list))
                        boundary_in_region =\
                            [ii for ii in np.arange(len(self.boundary_regions))\
                                if self.boundary_regions[ii][0] == i]
                        if boundary_in_region:
                            boundary_vertices = self.boundary_points[boundary_in_region] 
                            vertices.append(boundary_vertices)

            elif not identify_outside_vertices:
                vertices.append(self.vor.vertices[region])
        
        if vertices is not None:
            vertices = np.concatenate((vertices))
            unique_vertices = set(tuple(row) for row in vertices)
            return np.asarray([list(row) for row in unique_vertices])
        else: 
            return vertices
        
    def identify_vertices_outside_bounds(self, region):
        """
        Identify vertices that sit outside the bounding region

        :param region: A list of the voronoi regions. Each list contains indices of the voronoi 
            vertices forming each Voronoi region.
        :type region: list

        :return: Returns a new list of voronoi regions with vertices outside
            the bounding region replaced with -2.
        """

        #outside = lambda lb, ub, x: (x < lb) + (x > ub)
        # Create a boolean mask for vertices outside the bounds
        region = np.array(region)
        region_vertices = self.vor.vertices[region]
        outside_mask = np.zeros(region_vertices.shape, dtype=bool)

        for col_index in range(self.ndim):
            lb, ub = self.bounds[col_index,0], self.bounds[col_index, 1]
            #vert_outside, = np.where(outside(lb, ub, region_vertices[:, col_index]))
            outside_mask[:, col_index] |= \
                (region_vertices[:, col_index] < lb) | (region_vertices[:, col_index] > ub)

        # Get the indices of vertices that are outside the bounds
        vert_outside = np.where(outside_mask.any(axis=1))[0]
        if len(vert_outside) > 0:
            outside_vert_index = [region[i] for i in vert_outside]
            region[vert_outside] = -2
        return region.tolist()

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
            region_vertices = self.vor.vertices[region]

        return region_vertices

    def snip_ridge_vertices(self, region_index, region_point_index, region_tuple):

        # Find the ridge vertices for the specified region
        region_dict = {x[0]: x[1] for x in region_tuple}
        
        # the voronoi points that are equidistant from the ridge that lies between them
        ridge_point_indices = np.argwhere(self.vor.ridge_points == region_point_index)[:, 0]
        
        # the vertices at the end of each ridge
        region_ridge_vertices = [self.vor.ridge_vertices[i] for i in ridge_point_indices]

        new_vertices = []
        for rv in region_ridge_vertices:
            urv = [region_dict.get(num) for num in rv]
            if len(urv) == 2: #2D Voronoi region
                u, v = np.argsort(urv)
                # and urv[v] > 0: # only one vertice is out of bounds - snip one end to the boundary hull
                if urv[u] == -2:
                    ray_end = self.vor.vertices[rv[u]]
                    ray_origin = self.vor.vertices[rv[v]]
                    norm_ray_direction =\
                        self.get_normal_ray_direction(ray_origin, ray_end)
                    new_vertice = \
                        self.find_boundary_hull_ray_crossings(norm_ray_direction, ray_origin)
                    # confirm new vertice is in given region
                    if new_vertice is not None and region_index in self.get_voronoi_region(new_vertice)[0]:
                        # confirm point is within boundary hull
                        if self.bhullD.find_simplex(new_vertice) >= 0:
                            new_vertices.append(new_vertice)
                # both vertices are out of bounds - snip both ends to the boundary hull
                if urv[v] == -2:
                    ray_end = self.vor.vertices[rv[v]]
                    ray_origin = self.vor.vertices[rv[u]]
                    norm_ray_direction = \
                        self.get_normal_ray_direction(ray_origin, ray_end)
                    new_vertice =\
                        self.find_boundary_hull_ray_crossings(norm_ray_direction, ray_origin)
                    if new_vertice is not None and region_index in self.get_voronoi_region(new_vertice)[0]:
                        if self.bhullD.find_simplex(new_vertice) >= 0:
                            new_vertices.append(new_vertice)

            elif len(urv) > 2: #3D + Voronoi region
                nunbounded_vert = urv.count(-1) + urv.count(-2)
                if nunbounded_vert > 0 and nunbounded_vert < len(urv):

                    edges = [[rv[i], rv[(i+1) % len(rv)]] for i in range(len(rv))]
                    updated_edges = \
                        [[urv[i], urv[(i+1) % len(urv)]] for i in range(len(urv))]
                    unbounded_edges =\
                        [[i, edge] for i, edge in enumerate(updated_edges) if -2 in edge]
                    for i, ev in unbounded_edges:
                        u, v = np.argsort(ev)
                        if ev[u] == -2: # and ev[v] > 0:
                            ray_end = self.vor.vertices[edges[i][u]]
                            ray_origin = self.vor.vertices[edges[i][v]]
                            norm_ray_direction = \
                                self.get_normal_ray_direction(ray_origin, ray_end)
                            new_vertice = \
                                self.find_boundary_hull_ray_crossings(norm_ray_direction, ray_origin)
                            if new_vertice is not None and region_index in self.get_voronoi_region(new_vertice)[0]:
                                if self.bhullD.find_simplex(new_vertice) >= 0:
                                            new_vertices.append(new_vertice)
                        if ev[v] == -2: # and ev[v] > 0:
                            ray_end = self.vor.vertices[edges[i][v]]
                            ray_origin = self.vor.vertices[edges[i][u]]
                            norm_ray_direction = \
                                self.get_normal_ray_direction(ray_origin, ray_end)
                            new_vertice = \
                                self.find_boundary_hull_ray_crossings(norm_ray_direction, ray_origin)
                            if new_vertice is not None and region_index in self.get_voronoi_region(new_vertice)[0]:
                                if self.bhullD.find_simplex(new_vertice) >= 0:
                                            new_vertices.append(new_vertice)

        return np.asarray(new_vertices)

    def get_normal_ray_direction(self, ray_origin, ray_end):
        ray_direction = ray_end - ray_origin
        return ray_direction / np.linalg.norm(ray_direction)
        
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
        """Find the vertex that has the greatest distance from the cell centroid."""

        self.raise_if_invalid_region_index(region_index)
        vertices = self.get_region_vertices(region_index,\
            identify_outside_vertices=identify_outside_vertices)
        if vertices is not None:
            centroid = self.get_region_seed(region_index)
            distances = np.linalg.norm(vertices - centroid, axis=1)
            furthest_vertex_index = np.argmax(distances)
        else:
            furthest_vertex_index = None
        return vertices, furthest_vertex_index

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
        """Process a set of additional points.
        
        Voronoi has a built in function to add points 
            -- self.vor.add_points(points, restart=True).
        However, 'incremental` must be set to True to use the built-in add_points() 
        method and is very slow. Qhull throws an error for dim>2 when 
        incremental=True and restart=False. This class method, which rebuilds 
        'manually` is faster.
        """
        from scipy.spatial import Voronoi 
        if not isinstance(points, np.ndarray):
            raise TypeError("Input to add_points must be a NumPy array.")
        points = np.atleast_2d(points)
        
        if not points.shape[-1] == self._all_points.shape[-1]:
            raise ValueError(f"Points in add_points have a different dimension"
                             " ({points.shape[-1]}) than points in voronoi"
                             " tessellation ({self._all_points.shape[-1]})")
        
        points = self.remove_invalid_rows(points)
        if points.size == 0:
            logger.warning("All input points were NaN or Inf."
                           " No new points added to voronoi tessellation.")
            return

        # make sure all new points are unique
        all_points = np.vstack((self._all_points, points))
        unique_points = set(tuple(row) for row in all_points)
        self._all_points = np.asarray([list(row) for row in unique_points])
        self.vor = Voronoi(self._all_points)

    def raise_if_invalid_region_index(self, region_index):
        if region_index > len(self.vor.regions) or region_index < 0:
            raise ValueError('Invalid region index. Index must be in (0, nregions]')


class KFoldCrossValidation:
    def __init__(self, nsplits, group_kfold, interpolation_field, interpolation_values, 
                 scale, metric, target_field, param_names, surrogate_options):
        """Initialize the K-Fold Cross-Validation with a given surrogate model.
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

    def _check_npslits(self, training_params):
        if self.nsplits > training_params.shape[0]:
            self.nsplits = int(training_params.shape[0]/2.0)
            logger.warning("Input parameter \"nsplits\" can't be greater than " +
                           "the number of samples in KFoldCrosValidation. Reducing " +
                           "number of splits to approximately half the number of samples.")
        
    def perform_kfold_cv(self, training_params, training_data, groups):
        """
        Perform K-Fold Cross-Validation.

        :param X: Feature matrix (training samples).
        :type X: np.ndarray
        
        :param y: Target values (ground truth).
        :type y: np.ndarray
        
        :return: Returns the index of the sample with the greatest 
            prediction error and the corresponding error value.
            tuple (index_of_max_error, max_error)
        """
        self._check_npslits(training_params)
        self.groups = groups
        from sklearn.model_selection import GroupKFold, KFold
        from joblib import Parallel, delayed
        if self.group_kfold:
            assert self.groups is not None
            cv = GroupKFold(n_splits=self.nsplits)
            kf_results = Parallel(n_jobs=1)(
                delayed(self.evaluate_fold)(train_index, test_index, training_params, training_data, index)
                for index, (train_index, test_index) in enumerate(cv.split(training_params, training_data, self.groups))
            )
        else:
            cv = KFold(n_splits=self.nsplits, shuffle=True, random_state=1)
            kf_results = Parallel(n_jobs=1)(
                delayed(self.evaluate_fold)(train_index, test_index, training_params, training_data, index)
                for index, (train_index, test_index) in enumerate(cv.split(training_params))
            )
        # Convert the results to a dictionary
        kf = {k_idx: result for k_idx, result in enumerate(kf_results)}
        return kf

    def evaluate_fold(self, train_index, test_index, X, y, kfold_count):
        logger.info(f"\tEvaluating test '{self.metric}' error for surrogate for kfold cross validation set {kfold_count}..." +
                    "")
        info = self.extract_fold_info(train_index, test_index, X, y)
        train_eval_info, test_eval_info, X_test, y_test = info
        fold_surrogate = _fit_surrogate_model(train_eval_info, self.interpolation_field, 
                                              self.interpolation_values, test_eval_info, 
                                              self.target_field,
                                              "kfold_validation_surrogate.joblib", logger_on=False, 
                                              **self.surrogate_options)
        error = _get_surrogate_metric(fold_surrogate._latent_scores['test'], self.metric)
        logger.info(f"\t\terror = {error}")
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
        # Leave one out: create training and test sets
        logger.info(f"\tEvaluating test '{self.metric}' error for surrogate leaving out sample {index}" +
                    "")
        info = self.extract_loo_info(index, X, y)
        train_eval_info, test_eval_info, X_test, y_test = info
        fold_surrogate = _fit_surrogate_model(train_eval_info, self.interpolation_field, 
                                              self.interpolation_values, test_eval_info, 
                                              self.target_field,
                                              "kfold_validation_surrogate.joblib", 
                                               logger_on=False, 
                                               **self.surrogate_options)
        error = _get_surrogate_metric(fold_surrogate._latent_scores['test'], self.metric)
        logger.info(f"\t\terror = {error}")
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
