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
from dataclasses import dataclass, field, fields, replace

import numpy as np

from matcal.core.parameters import ParameterCollection
from matcal.dakota.dakota_interfaces import DakotaTermination


@dataclass
class ConvergenceSettings:
    """
    Stopping settings shared by the Dakota gradient calibration methods.

    :param relative: stop when the relative drop in the total objective
        between iterates is at most this value.
    :param absolute: stop when the total objective is at most this value.
    :param variable: stop when no bound-scaled parameter changes by more than
        this value between iterates. Unbounded parameters are not scaled, so
        their change is measured in their own units.
    :param max_function_evaluations: stop after this many evaluations.
    """

    relative: float | None = 1e-3
    absolute: float | None = None
    variable: float | None = None
    max_function_evaluations: int | None = 1000

    def is_active(self) -> bool:
        """:return: True if any setting is set."""
        return any(getattr(self, setting.name) is not None for setting in fields(self))


@dataclass(frozen=True)
class _DakotaKeywordMapping:
    """
    How one MatCal convergence setting is written as a Dakota keyword.

    :param keyword: the Dakota method keyword.
    :param to_dakota_factor: converts a MatCal value to the Dakota value. The
        integer default keeps integer settings such as
        ``max_function_evaluations`` integers in the deck.
    :param value_if_unset: written when the MatCal setting is not set.
    """

    keyword: str
    to_dakota_factor: float = 1
    value_if_unset: float | None = None

    def to_dakota_value(self, value: float) -> float:
        """:return: ``value`` in the units Dakota expects for ``keyword``."""
        return value * self.to_dakota_factor


@dataclass(frozen=True)
class _MethodSettings:
    """
    How one Dakota method receives MatCal's convergence settings.

    :param dakota_keywords: the MatCal settings Dakota handles for the method,
        by setting name. MatCal checks the others itself.
    :param fixed_lines: method keywords always written, with their values.
    :param dropped_keywords: default method keywords the method rejects.
    """

    dakota_keywords: Mapping[str, _DakotaKeywordMapping]
    fixed_lines: Mapping[str, str | float] = field(default_factory=dict)
    dropped_keywords: frozenset[str] = frozenset()

    def written_keywords(self) -> set[str]:
        """:return: every keyword this method's settings can write."""
        keywords = {mapping.keyword for mapping in self.dakota_keywords.values()}
        return keywords | set(self.fixed_lines)


_RELATIVE = _DakotaKeywordMapping("convergence_tolerance")
_MAX_EVALUATIONS = _DakotaKeywordMapping("max_function_evaluations")

_DEFAULT_METHOD_SETTINGS = _MethodSettings(
    {"relative": _RELATIVE, "max_function_evaluations": _MAX_EVALUATIONS}
)

ROL_OPTIONS_FILENAME = "matcal_rol_options.xml"

# On bound-scaled parameters ROL's automatic initial trust region radius can
# collapse when the projected gradient path reaches a bound, which ends the
# run on its step tolerance after one iteration. 0.1 is 10% of each range.
# Dakota leaves unbounded parameters unscaled, so when any parameter is
# unbounded MatCal leaves the initial radius to ROL.
ROL_INITIAL_RADIUS = 0.1

# DOT and CONMIN combine their relative test with absolute objective change
# tests that Dakota sets from the starting objective or does not expose, so
# they stopped early. They get a tight Dakota tolerance and MatCal checks the
# relative tolerance.
_TIGHT_DAKOTA_TOLERANCE = 1e-12
_MATCAL_RELATIVE_METHODS = ("dot_sqp", "dot_slp", "dot_mmfd", "conmin_mfd")

_METHOD_SETTINGS: dict[str, _MethodSettings] = {
    # NL2SOL minimizes half the sum of squared residuals; MatCal's total
    # objective for least squares methods is the full sum of squares.
    "nl2sol": _MethodSettings(
        {
            "relative": _RELATIVE,
            "max_function_evaluations": _MAX_EVALUATIONS,
            "absolute": _DakotaKeywordMapping(
                "absolute_conv_tol", to_dakota_factor=0.5
            ),
            "variable": _DakotaKeywordMapping("x_conv_tol"),
        }
    ),
    # Dakota's default ROL step tolerance is coarse for bound-scaled parameters.
    "rol": _MethodSettings(
        {"variable": _DakotaKeywordMapping("variable_tolerance", value_if_unset=1e-8)},
        fixed_lines={"options_file": f'"{ROL_OPTIONS_FILENAME}"'},
        dropped_keywords=frozenset(
            {"convergence_tolerance", "max_function_evaluations", "speculative"}
        ),
    ),
    **{
        method: _MethodSettings(
            {"max_function_evaluations": _MAX_EVALUATIONS},
            fixed_lines={"convergence_tolerance": _TIGHT_DAKOTA_TOLERANCE},
        )
        for method in _MATCAL_RELATIVE_METHODS
    },
}

_ALL_SETTING_KEYWORDS: frozenset[str] = frozenset().union(
    *(
        settings.written_keywords()
        for settings in [_DEFAULT_METHOD_SETTINGS, *_METHOD_SETTINGS.values()]
    )
)

# A single line search point with a tiny improvement must not end the run.
_RELATIVE_DROPS_REQUIRED = 2


def _method_settings(method: str) -> _MethodSettings:
    return _METHOD_SETTINGS.get(method, _DEFAULT_METHOD_SETTINGS)


def dakota_setting_lines(
    method: str, settings: ConvergenceSettings
) -> dict[str, float | int]:
    """
    :return: the Dakota method keywords and values for the settings that
        Dakota handles for ``method``.
    """
    lines = {}
    for name, mapping in _method_settings(method).dakota_keywords.items():
        value = getattr(settings, name)
        value = mapping.value_if_unset if value is None else value
        if value is not None:
            lines[mapping.keyword] = mapping.to_dakota_value(value)
    return lines


def method_specific_lines(method: str) -> dict[str, str | float]:
    """
    :return: extra Dakota method keywords MatCal always writes for ``method``.
    """
    return dict(_method_settings(method).fixed_lines)


def rol_options_xml(initial_radius: float | None) -> str:
    """
    :param initial_radius: the trust region radius for ROL's first iteration,
        or None to let ROL choose it.
    :return: a ROL parameter list for Dakota's ``options_file`` keyword.
    """
    if initial_radius is None:
        return '<ParameterList name="Inputs">\n</ParameterList>\n'
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
    handled_by_dakota = _method_settings(method).dakota_keywords
    return replace(settings, **{name: None for name in handled_by_dakota})


def keywords_to_remove(method: str, settings: ConvergenceSettings) -> set[str]:
    """
    :return: method keywords that must not appear in the Dakota input for
        ``method``: Dakota keywords of other methods or unset settings, and
        keywords ``method`` does not accept.
    """
    written = set(dakota_setting_lines(method, settings)) | set(
        method_specific_lines(method)
    )
    dropped = _method_settings(method).dropped_keywords
    return (set(_ALL_SETTING_KEYWORDS) - written) | set(dropped)


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
    """
    :return: ``values`` mapped to [0, 1] by each parameter's bounds. Values of
        parameters with an infinite bound are returned unchanged, as Dakota
        leaves such variables unscaled.
    """
    names = parameter_collection.get_item_names()
    lower = np.array([parameter_collection[name].get_lower_bound() for name in names])
    upper = np.array([parameter_collection[name].get_upper_bound() for name in names])
    width = upper - lower
    scalable = np.isfinite(width) & (width > 0)
    safe_width = np.where(scalable, width, 1.0)
    safe_lower = np.where(scalable, lower, 0.0)
    return np.where(scalable, (values - safe_lower) / safe_width, values)


def total_objective(dakota_functions: np.ndarray, is_residual: bool) -> float:
    """
    :return: MatCal's total objective for the values returned to Dakota: the
        sum of squared residuals for least squares methods, otherwise the sum
        of the objectives.
    """
    values = np.asarray(dakota_functions, dtype=float)
    return float(np.sum(values**2) if is_residual else np.sum(values))


def _matcal_stop(success: bool, message: str) -> DakotaTermination:
    outcome = "Converged" if success else "Stopped"
    return DakotaTermination(success, "MatCal", f"{outcome} (MatCal check): {message}")


class MatCalConvergenceReached(Exception):
    """Raised from the Dakota callback to stop a calibration MatCal judged done."""

    def __init__(self, termination: DakotaTermination) -> None:
        super().__init__(termination.message)
        self.termination = termination


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
    ) -> DakotaTermination | None:
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
    ) -> DakotaTermination | None:
        reason = self._check_absolute(objective)
        if reason is None and self._best_objective is not None:
            reason = self._check_relative(objective) or self._check_variable(
                scaled_center
            )
        return reason

    def _check_evaluations(self, evaluations: int) -> DakotaTermination | None:
        limit = self._settings.max_function_evaluations
        if limit is None or evaluations < limit:
            return None
        return _matcal_stop(False, f"maximum function evaluations ({limit}) reached")

    def _check_absolute(self, objective: float) -> DakotaTermination | None:
        tolerance = self._settings.absolute
        if tolerance is None or objective > tolerance:
            return None
        return _matcal_stop(
            True,
            f"absolute convergence: total objective {objective:g} <= {tolerance:g}",
        )

    def _check_relative(self, objective: float) -> DakotaTermination | None:
        tolerance, previous = self._settings.relative, self._best_objective
        if tolerance is None or previous is None or previous == 0.0:
            return None
        drop = (previous - objective) / abs(previous)
        self._small_relative_drops = (
            self._small_relative_drops + 1 if drop <= tolerance else 0
        )
        if self._small_relative_drops < _RELATIVE_DROPS_REQUIRED:
            return None
        return _matcal_stop(
            True,
            f"relative convergence: objective drop {drop:g} <= {tolerance:g} "
            f"for {_RELATIVE_DROPS_REQUIRED} consecutive iterates",
        )

    def _check_variable(self, scaled_center: np.ndarray) -> DakotaTermination | None:
        tolerance, previous = self._settings.variable, self._best_center
        if tolerance is None or previous is None:
            return None
        change = float(np.max(np.abs(scaled_center - previous)))
        if not np.isfinite(change) or change > tolerance:
            return None
        return _matcal_stop(
            True,
            f"variable convergence: scaled parameter change {change:g} "
            f"<= {tolerance:g}",
        )
