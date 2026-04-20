"""
Output-related SIERRA input-deck blocks for MatCal-generated SIERRA/SM decks.

Includes:
- User outputs and user variables
- Nonlocal averaging helpers
- Results output (exodus) and heartbeat output
- Solution termination and adaptive time stepping
"""

from matcal.core.input_file_writer import InputFileLine
from matcal.core.logger import initialize_matcal_logger
from matcal.core.utilities import check_value_is_nonempty_str

from .blocks_base import _BaseSierraInputFileBlock

logger = initialize_matcal_logger(__name__)


class SolidMechanicsUserOutput(_BaseSierraInputFileBlock):
    type = "user output"
    required_keys = []
    default_values = {}

    def __init__(
            self, output_name, mesh_entity_name=None, mesh_entity=None, compute_interval="every step"
        ):
        super().__init__(name=output_name)

        if mesh_entity_name is not None:
            if mesh_entity_name.lower().strip() == "include all blocks":
                mesh_entity_line = InputFileLine(mesh_entity_name)
            else:
                mesh_entity_line = InputFileLine(mesh_entity, mesh_entity_name)
            self.add_line(mesh_entity_line)
            self.set_print_name(False)
            self.set_print_title()

        compute_interval_line = InputFileLine("compute", compute_interval)
        compute_interval_line.set_symbol("at")
        self.add_line(compute_interval_line)

    def add_compute_global_as_function(self, name, function_name):
        self.add_compute_global(name, "as", "function", function_name)

    def add_compute_global_from_expression(self, name, expression):
        self.add_compute_global(name, "from", "expression", '"', expression, '"')

    def add_compute_global_from_nodal_field(self, name, field, calculation="average"):
        self.add_compute_global_from_field(name, field, "nodal", calculation)

    def add_compute_global_from_element_field(self, name, field, calculation="average"):
        self.add_compute_global_from_field(name, field, "element", calculation)

    def add_compute_global_from_field(self, name, field, source_entity, calculation="average"):
        self.add_compute_global(name, "as", calculation, "of", source_entity, field)

    def add_compute_global(self, name, *args):
        global_line = InputFileLine("compute", "global", name, *args, name="global " + name)
        global_line.suppress_symbol()
        self.add_line(global_line)

    def add_compute_element_from_element(
            self, name, source_name, calculation="volume weighted average", replace=False
        ):
        self.add_compute_element(name, "as", calculation, "of", "element", source_name)

    def add_compute_element(self, name, *args):
        element_line = InputFileLine("compute", "element", name, *args, name="element " + name)
        element_line.suppress_symbol()
        self.add_line(element_line, replace=True)

    def add_compute_element_as_function(self, name, function_name):
        self.add_compute_element(name, "as", "function", function_name)

    def add_nodal_variable_transformation(
            self, variable, transformed_name, target_coordinate_system
        ):
        transform_line = InputFileLine(
            "transform nodal variable",
            variable,
            "to coordinate system",
            target_coordinate_system,
            "as",
            transformed_name,
            name=transformed_name,
        )
        transform_line.suppress_symbol()
        self.add_line(transform_line)


class SolidMechanicsUserVariable(_BaseSierraInputFileBlock):
    type = "user variable"
    required_keys = ["type", "initial value"]
    default_values = {}

    def __init__(self, name, var_storage_location, var_type, *initial_values):
        super().__init__(name=name)
        self.set_print_name()
        self.set_print_title()

        var_type_line = InputFileLine(
            "type",
            var_storage_location,
            var_type,
            "length",
            "=",
            len(initial_values),
        )
        self.add_line(var_type_line)

        if var_storage_location == "global":
            self.add_global_operator()

        initial_value_line = InputFileLine("initial value", *initial_values)
        self.add_line(initial_value_line)

    def add_blocks(self, *blocks):
        blocks_line = InputFileLine("block", *blocks)
        self.add_line(blocks_line)

    def add_global_operator(self, operator="max"):
        operator_line = InputFileLine("global operator", operator)
        self.add_line(operator_line, replace=True)


class SolidMechanicsNonlocalAverage(_BaseSierraInputFileBlock):
    type = "nonlocal average"
    required_keys = ["source variable", "target_variable", "radius", "distance algorithm"]

    def __init__(
            self, source_variable, target_variable, radius, distance_algorithm="euclidean_graph"
        ):
        super().__init__()
        self.set_print_name(False)

        source_var_line = InputFileLine(self.required_keys[0], "element", source_variable)
        self.add_line(source_var_line)

        target_var_line = InputFileLine(self.required_keys[1], "element", target_variable)
        self.add_line(target_var_line)

        radius_line = InputFileLine(self.required_keys[2], radius)
        self.add_line(radius_line)

        distance_algorithm_line = InputFileLine(self.required_keys[3], distance_algorithm)
        self.add_line(distance_algorithm_line)


class SolidMechanicsNonlocalDamageAverage(SolidMechanicsNonlocalAverage):
    type = "nonlocal average"
    required_keys = ["source variable", "target_variable", "radius", "distance algorithm"]
    default_values = {}

    def __init__(self, radius, distance_algorithm="euclidean_graph"):
        super().__init__(
            "damage_increment", "nonlocal_damage_increment", radius, distance_algorithm
        )


class _SolidMechanicsOutputStepIncrement(_BaseSierraInputFileBlock):
    def __init__(self, output_step_increment, **kwargs):
        super().__init__(**kwargs)
        self.set_output_step_increment(output_step_increment)

    def set_output_step_increment(self, output_step_increment):
        output_increment_line = InputFileLine(
            "at step", 0, "," "increment", "=", output_step_increment
        )
        output_increment_line.suppress_symbol()
        self.add_line(output_increment_line, replace=True)


class _SolidMechanicsBaseOutput(_BaseSierraInputFileBlock):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.has_output_variables = False

    def add_output_variable(self, variable_scope, variable_name, save_as_name=None):
        new_line_args = [variable_scope, variable_name]
        if save_as_name is not None:
            check_value_is_nonempty_str(save_as_name, "save_as_name")
            new_line_args += ["as", save_as_name]

        name = self._get_line_name(variable_scope, variable_name, save_as_name)
        new_output_line = InputFileLine(*new_line_args, name=name)
        new_output_line.suppress_symbol()
        self.add_line(new_output_line)
        self.has_output_variables = True

    @staticmethod
    def _get_line_name(variable_scope, variable_name, save_as_name):
        name = variable_scope + " " + variable_name
        if save_as_name is not None:
            name += " " + save_as_name
        return name

    def add_global_output(self, variable_name, save_as_name=None):
        self.add_output_variable("global", variable_name, save_as_name)

    def has_global_output(self, variable_name, save_as_name=None):
        line_name = self._get_line_name("global", variable_name, save_as_name)
        return line_name in self.lines

    def has_element_output(self, variable_name, save_as_name=None):
        line_name = self._get_line_name("element", variable_name, save_as_name)
        return line_name in self.lines

    def get_global_outputs(self):
        global_lines = []
        for line_name in self.lines:
            if "global" in line_name:
                global_lines.append(self.lines[line_name])
        return global_lines


class SolidMechanicsResultsOutput(_SolidMechanicsOutputStepIncrement, _SolidMechanicsBaseOutput):
    type = "results output"
    required_keys = ["database name", "database type", "at step"]
    default_values = {required_keys[1]: "exodusII"}

    nodal_header = "nodal"
    element_header = "element"

    def __init__(
        self, output_step_increment, results_output_file="./results/results.e", 
        name="general_exodus_output"
    ):
        super().__init__(name=name, output_step_increment=output_step_increment)

        database_name_line = InputFileLine(self.required_keys[0], results_output_file)
        self.add_line(database_name_line)

    def add_nodal_output(self, variable_name, save_as_name=None):
        self.add_output_variable(self.nodal_header, variable_name, save_as_name)

    def add_element_output(self, variable_name, save_as_name=None):
        self.add_output_variable(self.element_header, variable_name, save_as_name)

    def add_include_surface(self, *surface_names):
        include_surf_line = InputFileLine("include", *surface_names)
        self.add_line(include_surf_line)

    def add_exclude_blocks(self, *blocks):
        exclude_blocks_line = InputFileLine("exclude", *blocks)
        self.add_line(exclude_blocks_line)

    def set_output_exposed_surface(self, output_exposed_surface=True):
        output_surface_key = "output mesh"
        if output_exposed_surface and output_surface_key not in self._lines:
            output_line = InputFileLine(output_surface_key, "exposed surface")
            self.add_line(output_line)
        elif (not output_exposed_surface) and output_surface_key in self._lines:
            self._lines.pop(output_surface_key)

    def _get_element_variable_line_name(self, element_variable_name, save_as_name=None):
        return _SolidMechanicsBaseOutput._get_line_name(
            self.element_header, element_variable_name, save_as_name
        )

    def _get_nodal_variable_line_name(self, nodal_variable_name, save_as_name=None):
        return _SolidMechanicsBaseOutput._get_line_name(
            self.nodal_header, nodal_variable_name, save_as_name
        )


class SolidMechanicsHeartbeatOutput(_SolidMechanicsOutputStepIncrement, _SolidMechanicsBaseOutput):
    type = "heartbeat output"
    required_keys = ["stream name"]
    default_values = {"labels": "off", "legend": "on", "precision": 16}

    def __init__(self, output_step_increment, *output_vars, filename="./results.csv"):
        super().__init__(name="csv_results_out", output_step_increment=output_step_increment)

        timestamp_line = InputFileLine("timestamp", "format", "''")
        timestamp_line.suppress_symbol()
        self.add_line(timestamp_line)

        stream_line = InputFileLine(self.required_keys[0], filename)
        self.add_line(stream_line)

        for var in output_vars:
            self.add_global_output(var)


class SolidMechanicsSolutionTermination(_BaseSierraInputFileBlock):
    type = "solution termination"
    required_keys = ["terminate type"]
    default_values = {"terminate type": "entire_run"}

    def add_global_termination_criteria(self, global_variable, value, operator="<"):
        termination_line = InputFileLine(
            "terminate",
            "global",
            global_variable,
            operator,
            value,
            name=f"global {global_variable}",
        )
        termination_line.suppress_symbol()
        self.add_line(termination_line, replace=True)


class SolidMechanicsAdaptiveTimeStepping(_BaseSierraInputFileBlock):
    type = "adaptive time stepping"
    required_keys = ["minimum multiplier", "maximum multiplier"]
    default_values = {"maximum failure cutbacks": 15}

    def __init__(self, minimum_multiplier=1e-8, maximum_multiplier=1):
        super().__init__()
        self.set_print_name(False)
        self.set_minimum_multiplier(minimum_multiplier)
        self.set_maximum_multiplier(maximum_multiplier)

    def set_cutback_factor(self, cutback_factor=0.5):
        self.add_line(InputFileLine("cutback factor", cutback_factor), replace=True)

    def set_growth_factor(self, growth_factor=1.5):
        self.add_line(InputFileLine("growth factor", growth_factor), replace=True)

    def set_minimum_multiplier(self, minimum_multiplier=1e-6):
        self.add_line(InputFileLine(self.required_keys[0], minimum_multiplier), replace=True)

    def set_maximum_multiplier(self, maximum_multiplier=1):
        self.add_line(InputFileLine(self.required_keys[1], maximum_multiplier), replace=True)

    def set_iteration_target(self, target_iterations=75, window=5):
        self.add_line(InputFileLine("target iterations", target_iterations), replace=True)
        self.add_line(InputFileLine("iteration window", window), replace=True)

    def set_adaptive_time_stepping_method(self, method):
        self.add_line(InputFileLine("method", method), replace=True)