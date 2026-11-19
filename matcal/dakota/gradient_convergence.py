"""
Shared convergence settings for MatCal's Dakota gradient calibrations.

Each setting has one MatCal meaning for every gradient method. When a method
has a Dakota keyword for a setting, the value is translated to that keyword.
Otherwise MatCal checks the setting itself on each batch Dakota requests: the
first parameter set in a batch is the point whose finite difference stencil
follows it, and a batch center that improves the best total objective so far
is an iterate.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, replace

import numpy as np

from matcal.core.parameters import ParameterCollection


@dataclass
class ConvergenceSettings:
    """
    Stopping settings shared by the Dakota gradient calibration methods.

    :param relative: stop when the relative drop in the total objective
        between iterates is at most this value.
    :param absolute: stop when the total objective is at most this value.
    :param variable: stop when no bound-scaled parameter changes by more than
        this value between iterates.
    :param max_function_evaluations: stop after this many evaluations.
    """

    relative: float | None = 1e-3
    absolute: float | None = None
    variable: float | None = None
    max_function_evaluations: int | None = 1000

    def is_active(self) -> bool:
        """:return: True if any setting is set."""
        return any(getattr(self, field.name) is not None for field in fields(self))


@dataclass(frozen=True)
class _NativeKeyword:
    keyword: str
    scale: float = 1.0
    default: float | None = None

    def translate(self, value: float) -> float | int:
        return value if self.scale == 1.0 else value * self.scale


_COMMON_KEYWORDS: dict[str, _NativeKeyword] = {
    "relative": _NativeKeyword("convergence_tolerance"),
    "max_function_evaluations": _NativeKeyword("max_function_evaluations"),
}

# NL2SOL minimizes half the sum of squared residuals; MatCal's total
# objective for least squares methods is the full sum of squares.
_NATIVE_KEYWORDS: dict[str, dict[str, _NativeKeyword]] = {
    "nl2sol": {
        **_COMMON_KEYWORDS,
        "absolute": _NativeKeyword("absolute_conv_tol", 0.5),
        "variable": _NativeKeyword("x_conv_tol"),
    },
    # Dakota's default ROL step tolerance is coarse for bound-scaled parameters.
    "rol": {"variable": _NativeKeyword("variable_tolerance", default=1e-8)},
}

# DOT and CONMIN combine their relative test with absolute objective change
# tests that Dakota sets from the starting objective or does not expose, so
# they stopped early. They get a tight native tolerance and MatCal checks the
# relative tolerance.
_TIGHT_NATIVE_TOLERANCE = 1e-12
_MATCAL_RELATIVE_METHODS = ("dot_sqp", "dot_slp", "dot_mmfd", "conmin_mfd")
for _method in _MATCAL_RELATIVE_METHODS:
    _NATIVE_KEYWORDS[_method] = {
        "max_function_evaluations": _COMMON_KEYWORDS["max_function_evaluations"]
    }

_UNSUPPORTED_KEYWORDS: dict[str, frozenset[str]] = {
    "rol": frozenset(
        {"convergence_tolerance", "max_function_evaluations", "speculative"}
    ),
}

ROL_OPTIONS_FILENAME = "matcal_rol_options.xml"

# On bound-scaled parameters ROL's automatic initial trust region radius can
# collapse when the projected gradient path reaches a bound, which ends the
# run on its step tolerance after one iteration. 0.1 is 10% of each range.
ROL_INITIAL_RADIUS = 0.1

_METHOD_SPECIFIC_LINES: dict[str, dict[str, str | float]] = {
    "rol": {"options_file": f'"{ROL_OPTIONS_FILENAME}"'},
    **{
        method: {"convergence_tolerance": _TIGHT_NATIVE_TOLERANCE}
        for method in _MATCAL_RELATIVE_METHODS
    },
}

# A single line search point with a tiny improvement must not end the run.
_RELATIVE_DROPS_REQUIRED = 2

_ALL_NATIVE_KEYWORDS: frozenset[str] = frozenset(
    [
        native.keyword
        for keywords in [_COMMON_KEYWORDS, *_NATIVE_KEYWORDS.values()]
        for native in keywords.values()
    ]
    + [keyword for lines in _METHOD_SPECIFIC_LINES.values() for keyword in lines]
)


def _native_keywords(method: str) -> dict[str, _NativeKeyword]:
    return _NATIVE_KEYWORDS.get(method, _COMMON_KEYWORDS)


def native_lines(method: str, settings: ConvergenceSettings) -> dict[str, float | int]:
    """
    :return: the Dakota method keywords and values for the settings that
        ``method`` supports natively.
    """
    lines = {}
    for name, native in _native_keywords(method).items():
        value = getattr(settings, name)
        value = native.default if value is None else value
        if value is not None:
            lines[native.keyword] = native.translate(value)
    return lines


def method_specific_lines(method: str) -> dict[str, str | float]:
    """
    :return: extra Dakota method keywords MatCal always writes for ``method``.
    """
    return dict(_METHOD_SPECIFIC_LINES.get(method, {}))


def rol_options_xml(initial_radius: float) -> str:
    """
    :param initial_radius: the trust region radius for ROL's first iteration.
    :return: a ROL parameter list for Dakota's ``options_file`` keyword.
    """
    return (
        '<ParameterList name="Inputs">\n'
        '  <ParameterList name="Step">\n'
        '    <ParameterList name="Trust Region">\n'
        f'      <Parameter name="Initial Radius" type="double" value="{initial_radius!r}"/>\n'
        "    </ParameterList>\n"
        "  </ParameterList>\n"
        "</ParameterList>\n"
    )


def matcal_checked_settings(
    method: str, settings: ConvergenceSettings
) -> ConvergenceSettings:
    """
    :return: a copy of ``settings`` keeping only the settings that ``method``
        has no Dakota keyword for, so MatCal must check them.
    """
    natively_supported = _native_keywords(method)
    return replace(settings, **{name: None for name in natively_supported})


def keywords_to_remove(method: str, settings: ConvergenceSettings) -> set[str]:
    """
    :return: method keywords that must not appear in the Dakota input for
        ``method``: native keywords of other methods or unset settings, and
        keywords ``method`` does not accept.
    """
    written = set(native_lines(method, settings)) | set(method_specific_lines(method))
    unsupported = _UNSUPPORTED_KEYWORDS.get(method, frozenset())
    return (set(_ALL_NATIVE_KEYWORDS) - written) | set(unsupported)


def batch_center(parameter_sets: Mapping | Sequence[Mapping]) -> np.ndarray:
    """
    :return: the continuous variables of the point whose finite difference
        stencil fills the rest of the Dakota batch.
    """
    first_set = (
        parameter_sets if isinstance(parameter_sets, Mapping) else parameter_sets[0]
    )
    return np.asarray(first_set["cv"], dtype=float)


def scale_to_bounds(
    values: np.ndarray, parameter_collection: ParameterCollection
) -> np.ndarray:
    """:return: ``values`` mapped to [0, 1] by each parameter's bounds."""
    names = parameter_collection.get_item_names()
    lower = np.array([parameter_collection[name].get_lower_bound() for name in names])
    upper = np.array([parameter_collection[name].get_upper_bound() for name in names])
    return (values - lower) / (upper - lower)


def total_objective(dakota_functions: np.ndarray, is_residual: bool) -> float:
    """
    :return: MatCal's total objective for the values returned to Dakota: the
        sum of squared residuals for least squares methods, otherwise the sum
        of the objectives.
    """
    values = np.asarray(dakota_functions, dtype=float)
    return float(np.sum(values**2) if is_residual else np.sum(values))


@dataclass(frozen=True)
class ConvergenceReason:
    """
    Why a MatCal convergence check stopped a calibration.

    :param success: True for a tolerance, False for an evaluation limit.
    :param message: a description of the criterion that was met.
    """

    success: bool
    message: str


class MatCalConvergenceReached(Exception):
    """Raised from the Dakota callback to stop a calibration MatCal judged done."""

    def __init__(self, reason: ConvergenceReason) -> None:
        super().__init__(reason.message)
        self.reason = reason


class MatCalConvergenceMonitor:
    """
    Checks the convergence settings a Dakota method has no keyword for.

    :param settings: the settings MatCal must check.
    """

    def __init__(self, settings: ConvergenceSettings) -> None:
        self._settings = settings
        self._best_objective: float | None = None
        self._best_center: np.ndarray | None = None
        self._small_relative_drops = 0

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self._settings!r}, "
            f"best_objective={self._best_objective})"
        )

    def update(
        self, scaled_center: np.ndarray, objective: float, evaluations: int
    ) -> ConvergenceReason | None:
        """
        Record a batch center and check the settings.

        :param scaled_center: the batch center scaled to the parameter bounds.
        :param objective: MatCal's total objective at the center.
        :param evaluations: the number of evaluations so far.
        :return: the reason to stop, or None to continue.
        """
        reason = self._check_evaluations(evaluations)
        if reason is None and self._is_improvement(objective):
            reason = self._check_iterate(scaled_center, objective)
            self._best_objective, self._best_center = objective, scaled_center
        return reason

    def _is_improvement(self, objective: float) -> bool:
        return self._best_objective is None or objective < self._best_objective

    def _check_iterate(
        self, scaled_center: np.ndarray, objective: float
    ) -> ConvergenceReason | None:
        reason = self._check_absolute(objective)
        if reason is None and self._best_objective is not None:
            reason = self._check_relative(objective) or self._check_variable(
                scaled_center
            )
        return reason

    def _check_evaluations(self, evaluations: int) -> ConvergenceReason | None:
        limit = self._settings.max_function_evaluations
        if limit is None or evaluations < limit:
            return None
        return ConvergenceReason(
            False, f"maximum function evaluations ({limit}) reached"
        )

    def _check_absolute(self, objective: float) -> ConvergenceReason | None:
        tolerance = self._settings.absolute
        if tolerance is None or objective > tolerance:
            return None
        return ConvergenceReason(
            True,
            f"absolute convergence: total objective {objective:g} <= {tolerance:g}",
        )

    def _check_relative(self, objective: float) -> ConvergenceReason | None:
        tolerance, previous = self._settings.relative, self._best_objective
        if tolerance is None or previous is None or previous == 0.0:
            return None
        drop = (previous - objective) / abs(previous)
        self._small_relative_drops = (
            self._small_relative_drops + 1 if drop <= tolerance else 0
        )
        if self._small_relative_drops < _RELATIVE_DROPS_REQUIRED:
            return None
        return ConvergenceReason(
            True,
            f"relative convergence: objective drop {drop:g} <= {tolerance:g} "
            f"for {_RELATIVE_DROPS_REQUIRED} consecutive iterates",
        )

    def _check_variable(self, scaled_center: np.ndarray) -> ConvergenceReason | None:
        tolerance, previous = self._settings.variable, self._best_center
        if tolerance is None or previous is None:
            return None
        change = float(np.max(np.abs(scaled_center - previous)))
        if change > tolerance:
            return None
        return ConvergenceReason(
            True,
            f"variable convergence: scaled parameter change {change:g} "
            f"<= {tolerance:g}",
        )
