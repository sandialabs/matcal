"""
Core SIERRA input-deck building blocks used by MatCal.

This module contains:
- Base block class for SIERRA-typed blocks
- Global definitions block
- Function blocks (analytic + piecewise linear)
- Physics naming helpers
- THERMAL_MATERIAL_NAME constant
"""

import numbers

from matcal.core.input_file_writer import (
    _BaseTypedInputFileBlock,
    InputFileLine,
    InputFileTable,
)
from matcal.core.logger import initialize_matcal_logger
from matcal.core.utilities import check_item_is_correct_type

logger = initialize_matcal_logger(__name__)


class _BaseSierraInputFileBlock(_BaseTypedInputFileBlock):
    def __init__(self, name=None, begin_end=True, **kwargs):
        super().__init__(name=name, begin_end=begin_end, **kwargs)
        self.set_print_name()
        self.set_print_title()


class SierraGlobalDefinitions(_BaseSierraInputFileBlock):
    type = "global_definitions"
    required_keys = []
    default_values = {
        "define direction X": ["vector", 1.0, 0.0, 0.0],
        "define direction Y": ["vector", 0.0, 1.0, 0.0],
        "define direction Z": ["vector", 0.0, 0.0, 1.0],
        "define point o": ["coordinates", 0.0, 0.0, 0.0],
        "define point x": ["coordinates", 1.0, 0.0, 0.0],
        "define point y": ["coordinates", 0.0, 1.0, 0.0],
        "define point z": ["coordinates", 0.0, 0.0, 1.0],
        "define coordinate system rectangular_coordinate_system rectangular": [
            "point",
            "o",
            "point",
            "z",
            "point x",
        ],
        "define axis cylindrical_axis": ["point", "o", "direction", "Y"],
        "define coordinate system cylindrical_coordinate_system cylindrical": [
            "point",
            "o",
            "point",
            "y",
            "point x",
        ],
    }

    def __init__(self):
        """
        Standard global definitions for MatCal-generated SIERRA input decks.
        """
        super().__init__(begin_end=False)
        self.set_print_title(False)
        self.set_print_name(False)
        self.set_symbol_for_lines("with")


class _BaseSierraFunction(_BaseSierraInputFileBlock):
    type = "function"
    required_keys = ["type"]
    default_values = {}

    def __init__(self, name, function_type):
        super().__init__(name=name, begin_end=True)
        function_type_line = InputFileLine("type", function_type)
        function_type_line.set_symbol("is")
        self.add_line(function_type_line)

    def scale_function(self, x=None, y=None):
        """
        Apply a scale factor to the function's independent or dependent variables.

        :param x: independent variable scale factor.
        :type x: float

        :param y: dependent variable scale factor.
        :type y: float
        """
        self._scale_function_column("x", x)
        self._scale_function_column("y", y)

    def _scale_function_column(self, col_name, val):
        line_name = col_name + " scale"
        if line_name in self._lines:
            self._lines.pop(line_name)
        if val is not None:
            check_item_is_correct_type(val, numbers.Real, "scale_factor", call_depth=1)
            line = InputFileLine(line_name, val)
            self.add_line(line)


class AnalyticSierraFunction(_BaseSierraFunction):
    required_keys = _BaseSierraFunction.required_keys + ["evaluate expression"]

    def __init__(self, name):
        """
        Analytical SIERRA function.
        """
        super().__init__(name, function_type="analytic")

    def add_expression_variable(self, name, *vals):
        """
        Add an expression variable derived from existing variables.
        """
        line = InputFileLine(f"expression variable: {name}", *vals, name=name)
        self.add_line(line)

    def add_evaluation_expression(self, expression):
        """
        Add an expression used to evaluate the function.
        """
        line = InputFileLine("evaluate expression", '"', expression, '"')
        line.set_symbol("=")
        self.add_line(line)


class PiecewiseLinearFunction(_BaseSierraFunction):
    def __init__(self, name, x_values, y_values):
        """
        Piecewise-linear SIERRA function.
        """
        super().__init__(name, function_type="piecewise linear")
        func_table = InputFileTable("values", 2)
        func_table.set_values(x_values, y_values)
        self.add_table(func_table)


class _PhysicsNames:
    coupled = "coupled"
    thermal = "thermal"
    solid_mechanics = "solid_mechanics"


def _get_default_thermal_region_name():
    return f"{_PhysicsNames.thermal}_region"


def _get_default_solid_mechanics_region_name():
    return f"{_PhysicsNames.solid_mechanics}_region"


def _get_default_solid_mechanics_procedure_name():
    return f"{_PhysicsNames.solid_mechanics}_procedure"


def _get_default_coupled_procedure_name():
    return f"{_PhysicsNames.coupled}_procedure"


THERMAL_MATERIAL_NAME = f"matcal_{_PhysicsNames.thermal}"