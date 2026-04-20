"""
Time control / time stepping blocks for MatCal-generated SIERRA input decks.
"""

from matcal.core.input_file_writer import InputFileLine

from .blocks_base import _BaseSierraInputFileBlock


class _BaseTimeParameters(_BaseSierraInputFileBlock):
    def __init__(self, region_name, time_increment):
        super().__init__(region_name)
        self.set_time_increment(time_increment)

    def set_time_increment(self, time_increment):
        """
        Set the time increment for this parameters block.
        """
        self.add_line(InputFileLine(self.required_keys[0], time_increment), replace=True)

    @property
    def time_increment(self):
        return self.get_line_value(self.required_keys[0])


class ThermalTimeParameters(_BaseTimeParameters):
    """
    Time parameters for SIERRA/Thermal Aria region.
    """

    type = "parameters for aria region"
    required_keys = ["initial time step size", "time step variation"]
    default_values = {"time step variation": "fixed"}


class SolidMechanicsTimeParameters(_BaseTimeParameters):
    """
    Time parameters for SIERRA/SM Adagio region.
    """

    type = "parameters for adagio region"
    required_keys = ["time increment"]
    default_values = {}


class TimeSteppingBlock(_BaseSierraInputFileBlock):
    type = "time stepping block"
    required_keys = ["start time"]
    default_values = {}

    def __init__(self, name, region_name, start_time, time_increment):
        super().__init__(name)
        self.set_start_time(start_time)
        self._time_parameters = SolidMechanicsTimeParameters(region_name, time_increment)
        self.add_subblock(self._time_parameters)

    def set_start_time(self, start_time):
        self.add_line(InputFileLine("start time", start_time), replace=True)

    def set_time_increment(self, time_increment):
        self._time_parameters.set_time_increment(time_increment)


class TimeControl(_BaseSierraInputFileBlock):
    type = "time control"
    required_keys = ["termination time"]
    default_values = {}

    def __init__(self, termination_time, *time_stepping_blocks):
        super().__init__()
        self.set_termination_time(termination_time)
        for time_stepping_block in time_stepping_blocks:
            self.add_subblock(time_stepping_block)
        self._print_name = False

    def set_termination_time(self, termination_time):
        self.add_line(InputFileLine("termination time", termination_time), replace=True)


class SolidMechanicsProcedure(_BaseSierraInputFileBlock):
    """
    Adagio procedure block including default time control setup.
    """

    type = "adagio procedure"
    required_keys = []
    default_values = {}

    def __init__(
        self,
        name,
        solid_mechanics_region,
        start_time,
        termination_time,
        time_steps,
        init_time_step_scale_factor=1e-3,
    ):
        super().__init__(name)
        self._solid_mechanics_region = solid_mechanics_region
        self._start_time = start_time
        self._termination_time = termination_time
        self._time_steps = time_steps
        self._init_time_step_scale_factor = init_time_step_scale_factor

        self._time_step = None
        self._small_time_step = None
        self._time_control_block = None

        self._set_time_step()
        self._set_small_time_step()
        self._init_elastic_time_step_block()
        self._init_load_time_stepping_block()
        self._add_time_control_block()

        self.add_subblock(solid_mechanics_region)

    def _set_time_step(self):
        self._time_step = (self._termination_time - self._start_time) / self._time_steps

    def _set_small_time_step(self):
        self._small_time_step = self._time_step * self._init_time_step_scale_factor

    def _init_elastic_time_step_block(self):
        self._elastic_time_step_block = TimeSteppingBlock(
            "elastic_init",
            self._solid_mechanics_region.name,
            self._start_time,
            self._small_time_step,
        )

    def _init_load_time_stepping_block(self):
        self._load_time_step_block = TimeSteppingBlock(
            "load",
            self._solid_mechanics_region.name,
            self._start_time + self._small_time_step,
            self._time_step,
        )

    def _add_time_control_block(self):
        self._time_control_block = TimeControl(
            self._termination_time,
            self._elastic_time_step_block,
            self._load_time_step_block,
        )
        self.add_subblock(self._time_control_block)

    def _update_time_params(self):
        self._set_time_step()
        self._set_small_time_step()
        self._elastic_time_step_block.set_start_time(self._start_time)
        self._elastic_time_step_block.set_time_increment(self._small_time_step)
        self._load_time_step_block.set_start_time(self._start_time + self._small_time_step)
        self._load_time_step_block.set_time_increment(self._time_step)
        self._time_control_block.set_termination_time(self._termination_time)

    def set_number_of_time_steps(self, time_steps):
        self._time_steps = time_steps
        self._update_time_params()

    def set_start_time(self, start_time):
        self._start_time = start_time
        self._update_time_params()

    def set_end_time(self, end_time):
        self._termination_time = end_time
        self._update_time_params()