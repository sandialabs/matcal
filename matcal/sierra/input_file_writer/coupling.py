"""
Coupling (Arpeggio) input-deck blocks for MatCal-generated SIERRA coupled simulations.

Includes:
- Coupling mode enums (_Coupling, _Failure)
- Transfer blocks
- Transient blocks (including optional nonlinear wrapping)
- Coupled system/initialize blocks
- Solution control and procedure blocks
"""

from collections import OrderedDict

from matcal.core.input_file_writer import InputFileLine

from .blocks_base import _BaseSierraInputFileBlock


class _Coupling:
    adiabatic = "adiabatic"
    staggered = "staggered"
    iterative = "iterative"


class _Failure:
    local_failure = "local"
    nonlocal_failure = "nonlocal"


class ArpeggioTransfer(_BaseSierraInputFileBlock):
    type = "transfer"
    required_keys = ["copy"]
    default_values = {}

    def __init__(self, name):
        super().__init__(name)

    def _set_mesh_entity_copy(self, mesh_entity, sending_region, receiving_region):
        copy_line = InputFileLine(
            "copy", "volume", mesh_entity, "from", sending_region, "to", receiving_region
        )
        copy_line.suppress_symbol()
        self.add_line(copy_line, replace=True)

    def set_element_copy_transfer(self, sending_region, receiving_region):
        self._set_mesh_entity_copy("elements", sending_region, receiving_region)

    def set_nodal_copy_transfer(self, sending_region, receiving_region):
        self._set_mesh_entity_copy("nodes", sending_region, receiving_region)

    def add_field_to_send(
        self, sending_field, receiving_field, sending_state="none", receiving_state="none"
    ):
        name = self.get_line_name(sending_field, receiving_field, sending_state, receiving_state)
        send_line = InputFileLine(
            "send",
            "field",
            sending_field,
            "state",
            sending_state,
            "to",
            receiving_field,
            "state",
            receiving_state,
            name=name,
        )
        send_line.suppress_symbol()
        self.add_line(send_line)

    @staticmethod
    def get_line_name(sending_field, receiving_field, sending_state="none", receiving_state="none"):
        return "_".join([sending_field, sending_state, receiving_field, receiving_state])

    def add_send_blocks(self, *blocks):
        send_blocks = " ".join(blocks)
        send_blocks_line = InputFileLine(
            "send", "block", send_blocks, "to", send_blocks, name="send_blocks"
        )
        send_blocks_line.suppress_symbol()
        self.add_line(send_blocks_line)


class CoupledTransientParameters(_BaseSierraInputFileBlock):
    type = "parameters for transient"
    required_keys = ["start time", "termination time"]
    default_values = {}

    def __init__(
        self, name, thermal_region_name, solid_mechanics_region_name, 
        start_time, termination_time, time_step
    ):
        super().__init__(name)

        # subblocks are time parameter blocks for each region; in the monolith these were
        # SolidMechanicsTimeParameters and ThermalTimeParameters. Here, we keep the interface
        # and allow the calling code to set both increments consistently by exposing set_time_increment.
        from .time_control import SolidMechanicsTimeParameters, ThermalTimeParameters  # local import to avoid cycles

        self._solid_mechanics_params = SolidMechanicsTimeParameters(solid_mechanics_region_name, time_step)
        self._thermal_params = ThermalTimeParameters(thermal_region_name, time_step)
        self.add_subblock(self._solid_mechanics_params)
        self.add_subblock(self._thermal_params)

        lines = {self.required_keys[0]: start_time, self.required_keys[1]: termination_time}
        self.add_lines_from_dictionary(lines)

    def set_start_time(self, start_time):
        self.lines[self.required_keys[0]].set(start_time)

    def set_termination_time(self, termination_time):
        self.lines[self.required_keys[1]].set(termination_time)

    def set_time_increment(self, time_increment):
        self._solid_mechanics_params.set_time_increment(time_increment)
        self._thermal_params.set_time_increment(time_increment)

    @property
    def start_time(self):
        return self.get_line_value(self.required_keys[0])

    @property
    def termination_time(self):
        return self.get_line_value(self.required_keys[1])

    @property
    def time_increment(self):
        sm_time_increment = self._solid_mechanics_params.time_increment
        thermal_time_increment = self._thermal_params.time_increment
        if sm_time_increment == thermal_time_increment:
            return sm_time_increment
        raise ValueError("Thermal and solid mechanics time increments are not equal.")


class NonlinearStep(_BaseSierraInputFileBlock):
    type = "nonlinear"
    required_keys = []
    default_values = {}

    def __init__(self, name, *lines):
        super().__init__(name)
        for line in lines:
            self.add_line(line)


class NonlinearParameters(_BaseSierraInputFileBlock):
    type = "parameters for nonlinear"
    required_keys = ["converged when"]
    default_values = {
        "converged when": (
            '"thermal_region.MaxInitialNonlinearResidual(0) < 1.0e-8 '
            ' || CURRENT_STEP > 20"'
        )
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_symbol_for_lines(None)


class CoupledTransient(_BaseSierraInputFileBlock):
    type = "transient"
    required_keys = []
    default_values = {}

    def __init__(self, name, solid_region, thermal_region, nonlinear_step_name=None):
        super().__init__(name)
        self._nonlinear_step_name = nonlinear_step_name
        self._advance_solid = InputFileLine("advance", solid_region, name="advance_solid")
        self._advance_thermal = InputFileLine("advance", thermal_region, name="advance_thermal")
        self._post_solid_lines = []
        self._post_thermal_lines = []

    def _create_transfer_line(self, transfer_name):
        transfer_line = InputFileLine("transfer", transfer_name, name=transfer_name)
        return transfer_line

    def add_transfer_post_solid_mechanics(self, transfer_name):
        self._post_solid_lines.append(self._create_transfer_line(transfer_name))

    def add_transfer_post_thermal(self, transfer_name):
        self._post_thermal_lines.append(self._create_transfer_line(transfer_name))

    def set_nonlinear_step_name(self, nonlinear_step_name):
        self._nonlinear_step_name = nonlinear_step_name

    def _setup_lines(self):
        self.reset_lines()

        self.add_line(self._advance_solid)
        for line in self._post_solid_lines:
            self.add_line(line)

        self.add_line(self._advance_thermal)
        for line in self._post_thermal_lines:
            self.add_line(line)

        self.set_symbol_for_lines(None)

        if self._nonlinear_step_name is not None:
            nonlinear_step_block = NonlinearStep(self._nonlinear_step_name, *self.lines.values())
            self._lines = OrderedDict()
            self.add_subblock(nonlinear_step_block, replace=True)

    def get_string(self):
        self._setup_lines()
        return super().get_string()


class CoupledSystem(_BaseSierraInputFileBlock):
    type = "system"
    required_keys = ["use initialize"]
    default_values = {}

    def __init__(self, name, initializer_name, *transients):
        super().__init__(name)

        initializer_line = InputFileLine("use initialize", initializer_name)
        initializer_line.suppress_symbol()
        self.add_line(initializer_line)

        for transient in transients:
            self.add_subblock(transient)


class CoupledInitialize(CoupledTransient):
    type = "initialize"

    def __init__(self, name, solid_region, thermal_region):
        super().__init__(name, solid_region, thermal_region)


class SolutionControl(_BaseSierraInputFileBlock):
    type = "solution control description"
    required_keys = ["use system"]
    default_values = {}

    def __init__(self, name, system, initializer, *transient_parameter_sets):
        super().__init__(name)

        use_system_line = InputFileLine("use system", system.name)
        use_system_line.suppress_symbol()
        self.add_line(use_system_line)

        self.add_subblock(system)
        self.add_subblock(initializer)

        for transient_parameter_set in transient_parameter_sets:
            self.add_subblock(transient_parameter_set)

    def set_transient_time_parameters(self, transient_name, start_time, end_time, time_step):
        transient_time_params = self.subblocks[transient_name]
        transient_time_params.set_start_time(start_time)
        transient_time_params.set_termination_time(end_time)
        transient_time_params.set_time_increment(time_step)


class Procedure(_BaseSierraInputFileBlock):
    type = "procedure"
    required_keys = []
    default_values = {}

    def __init__(self, solution_control_block, *transfers, name=None):
        # default name is defined in blocks_base helper; import locally to avoid cycles
        if name is None:
            from .blocks_base import _get_default_coupled_procedure_name

            name = _get_default_coupled_procedure_name()

        super().__init__(name)
        self._solution_control = solution_control_block
        self.add_subblock(solution_control_block)
        for transfer in transfers:
            self.add_subblock(transfer)