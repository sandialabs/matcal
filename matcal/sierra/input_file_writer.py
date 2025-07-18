from collections import OrderedDict
import numbers

from matcal.core.boundary_condition_calculators import (
    get_temperature_function_from_data_collection) 
from matcal.core.constants import TEMPERATURE_KEY, TIME_KEY, DISPLACEMENT_KEY
from matcal.core.input_file_writer import (_BaseTypedInputFileBlock, InputFileBlock,
                                           InputFileLine, InputFileTable)
from matcal.core.logger import initialize_matcal_logger
from matcal.core.utilities import (check_item_is_correct_type, 
                                   check_value_is_nonempty_str, 
                                   get_string_from_text_file)

    
logger = initialize_matcal_logger(__name__)


class _BaseSierraInputFileBlock(_BaseTypedInputFileBlock):

    def __init__(self, name=None, begin_end=True, **kwargs):
        super().__init__(name=name, begin_end=begin_end, **kwargs)
        self.set_print_name()
        self.set_print_title()
        

class SierraGlobalDefinitions(_BaseSierraInputFileBlock):
    type = "global_definitions"
    required_keys = [] 
    default_values = {"define direction X": ["vector", 1.0, 0.0, 0.0], 
                      "define direction Y": ["vector", 0.0, 1.0, 0.0], 
                      "define direction Z": [ "vector", 0.0, 0.0, 1.0], 
                      "define point o": [ "coordinates", 0.0, 0.0, 0.0], 
                      "define point x": [ "coordinates", 1.0, 0.0, 0.0], 
                      "define point y": [ "coordinates", 0.0, 1.0, 0.0], 
                      "define point z": [ "coordinates", 0.0, 0.0, 1.0], 
                      "define coordinate system rectangular_coordinate_system rectangular":
                      ["point", "o", "point", "z", "point x"], 
                      "define axis cylindrical_axis":["point", "o", "direction", "Y"], 
                      "define coordinate system cylindrical_coordinate_system cylindrical":
                      ["point", "o", "point", "y", "point x"]}
    
    def __init__(self):
        """Sets standard global definitions for SIERRA input decks.
        This includes the global X,Y,Z cartesian coordinates and
        the rectangular and cylindrical coordinate systems detailed in 
        :ref:`MatCal Generated SIERRA Standard Models`.
        More global definitions can be added with the
        :mod:`~matcal.core.input_file_writer` tools."""
        super().__init__(begin_end=False)
        self.set_print_title(False)
        self.set_print_name(False)
        self.set_symbol_for_lines("with")


class _BaseSierraFunction(_BaseSierraInputFileBlock):
    type = "function"
    required_keys = ["type"] 
    default_values = {}
   
    def __init__(self, name, function_type):
        super().__init__(name=name,  begin_end=True)
        function_type_line = InputFileLine("type", function_type)
        function_type_line.set_symbol("is")
        self.add_line(function_type_line)

    def scale_function(self, x=None, y=None):
        """Apply a scale factor to the function's
        independent or dependent variables.
        
        :param x: the independent variable scale factor.
        :type x: float
        
        :param y: the dependent variable scale factor.
        :type y: float
        """
        self._scale_function_column("x", x)
        self._scale_function_column("y", y)
    
    def _scale_function_column(self, col_name, val):
        line_name = col_name + " scale"
        if line_name in self._lines:
            self._lines.pop(line_name)
        if val is not None:
            check_item_is_correct_type(val, numbers.Real, 
                                       "SierraFunction.scale_function",
                                         "scale_factor")
            line = InputFileLine(line_name, val)
            self.add_line(line)
            

class AnalyticSierraFunction(_BaseSierraFunction):
    required_keys = _BaseSierraFunction.required_keys + ["evaluate expression"]
    def __init__(self, name):
        """An analytical SIERRA input file function.

        :param name: the function name
        :type name: str
        """
        super().__init__(name, function_type="analytic")

    def add_expression_variable(self, name, *vals):
        """Add a new expression variable derived from existing global variables.

        :param name: the name of the variable to be created for the expression.
        :type name: str

        :param vals: the components of a the statement from which the new expression 
            variable will be calculated.
        :type vals: list(str,float,int)
        """
        line = InputFileLine(f"expression variable: {name}", *vals, name=name)
        self.add_line(line)

    def add_evaluation_expression(self, expression):
        """
        Add a valid expression from which to calculate the function dependent
        variable. 

        :param expression: valid expression for the SIERRA analytical function. 
        :param expression: str
        """
        line = InputFileLine("evaluate expression", '"', expression, '"')
        line.set_symbol("=")
        self.add_line(line)


class PiecewiseLinearFunction(_BaseSierraFunction):

    def __init__(self, name, x_values, y_values):
        """A piecewise-linear SIERRA input file function.

        :param name: the function name
        :type name: str

        :param x_values: the independent variables values for the function
        :type x_values: list(float)

        :param y_values: the dependent variables values for the function
        :type y_values: list(float)
        """
        super().__init__(name, function_type="piecewise linear")
        func_table = InputFileTable("values", 2)
        func_table.set_values(x_values, y_values)
        self.add_table(func_table)


class _PhysicsNames:
    coupled = "coupled"
    thermal = "thermal"
    solid_mechanics = "solid_mechanics"


def get_default_thermal_region_name():
    return f"{_PhysicsNames.thermal}_region"


def get_default_solid_mechanics_region_name():
    return f"{_PhysicsNames.solid_mechanics}_region"


def get_default_solid_mechanics_procedure_name():
    return f"{_PhysicsNames.solid_mechanics}_procedure"


def get_default_coupled_procedure_name():
    return f"{_PhysicsNames.coupled}_procedure"


THERMAL_MATERIAL_NAME = f"matcal_{_PhysicsNames.thermal}"


class ThermalMaterial(_BaseSierraInputFileBlock):
    type = "aria material"
    required_keys = ["density", "thermal conductivity", "specific heat"] 
    default_values = {}

    def __init__(self, density, thermal_conductivity, 
                 specific_heat, name=THERMAL_MATERIAL_NAME):
        """
        Creates a basic aria material model with conduction when provided
        the appropriate material parameters.

        :param density: the material density
        :type density: float

        :param thermal_conductivity: the material thermal conductivity
        :type thermal_conductivity: float

        :param specific_heat: the material specific heat
        :type specific_heat: float
        """
        super().__init__(name)
        density_line = InputFileLine("density", "constant", "rho", "=", density)
        self.add_line(density_line)
        conductivity_line = InputFileLine("thermal conductivity", "constant", 
                             "K", "=", thermal_conductivity)
        self.add_line(conductivity_line)
        spec_heat_line = InputFileLine("specific heat", "constant", "cp", 
                                       "=", specific_heat)
        self.add_line(spec_heat_line)
        self.add_line(InputFileLine("heat conduction", "basic"))


class FetiSolver(_BaseSierraInputFileBlock):
    type = "feti equation solver"
    required_keys = []
    default_values = {}

    def __init__(self, name="feti"):
        """
        Creates an empty SIERRA Feti linear solver block. 
        Users can add options to it using the 
        tools in :mod:`matcal.core.input_file_writer`.
        """
        super().__init__(name)


class GdswSolver(_BaseSierraInputFileBlock):
    type = "gdsw equation solver"
    required_keys = []
    default_values = {}
    def __init__(self, name="gdsw"):
        """
        Creates an empty SIERRA GDSW linear solver block. 
        Users can add options to it using the 
        tools in :mod:`matcal.core.input_file_writer`.
        """
        super().__init__(name)


class TpetraSolver(_BaseSierraInputFileBlock):
    type = "tpetra equation solver"
    required_keys = []
    default_values = {}
    def __init__(self, name="tpetra"):
        """
        Creates a prepopulated SIERRA Tpetra linear solver block
        with settings specialized for thermal solves in Aria. 
        Users can add options to it using the 
        tools in :mod:`matcal.core.input_file_writer`.
        """
        super().__init__(name=name)
        preset_solver_block = InputFileBlock("preset solver", begin_end=True)
        preset_solver_options = {}
        preset_solver_options['convergence tolerance'] = 1e-10
        preset_solver_options['maximum iterations'] = 10000
        preset_solver_options['residual scaling'] = 'r0'
        preset_solver_options['solver type'] = "thermal_symmetric"
        preset_solver_block.add_lines_from_dictionary(preset_solver_options)
        self.add_subblock(preset_solver_block)


class _SectionNames:
    total_lagrange = "total_lagrange"
    uniform_gradient = "uniform_gradient"
    composite_tet = "composite_tet"


class TotalLagrangeSection(_BaseSierraInputFileBlock):
    """
    Creates a SIERRA total lagrange section block.
    Users can add options to it using the 
    tools in :mod:`matcal.core.input_file_writer`.
    """
    type = "total lagrange section"
    required_keys = []
    default_values = {}

    def __init__(self, name=_SectionNames.total_lagrange):
        super().__init__(name)
        self.add_line(InputFileLine("volume average J", "on"))

    def use_composite_tet(self, use_composite_tet=True):
        """
        Change the section to use the composite tet formulation 
        for the total lagrange section. 

        :param use_composite_tet: flag to turn the composite tet formulation 
            of or on.
        :type use_composite_tet: bool
        """
        if use_composite_tet:
            if "formulation" not in self._lines:
                self.add_line(InputFileLine("formulation", "composite_tet"), 
                              replace=True)
            self.set_name(_SectionNames.composite_tet)
        elif not use_composite_tet and "formulation" in self._lines:
            self._lines.pop("formulation")
            self.set_name(_SectionNames.total_lagrange)

    
class SolidSectionDefault(_BaseSierraInputFileBlock):
    type = "solid section"
    required_keys = []
    default_values = {"strain incrementation":"strongly_objective"}

    def __init__(self, name=_SectionNames.uniform_gradient):
        """
        Creates a SIERRA default section block.
        This does set the strain incrementation to \'strongly_objective\'.
        Users can add options to it using the 
        tools in :mod:`matcal.core.input_file_writer`.
        """
        super().__init__(name)


class _BaseTimeParameters(_BaseSierraInputFileBlock):

    def __init__(self, region_name, time_increment):
        super().__init__(region_name)
        self.set_time_increment(time_increment)

    def set_time_increment(self, time_increment):
        """
        Set the time increment for this parameters block. 

        :param time_increment: the time increment or time step for this parameters block
        :type time_increment: float
        """
        self.add_line(InputFileLine(self.required_keys[0], time_increment), replace=True)

    @property
    def time_increment(self):
        return self.get_line_value(self.required_keys[0])


class ThermalTimeParameters(_BaseTimeParameters):
    """
    Creates a time parameters block for a SIERRA/Thermal Aria region.

    :param region: The name of the Aria region.
    :type region: str
    :param time_increment: The initial time step size for the region.
    :type time_increment: float
    """
    type = "parameters for aria region"
    required_keys = ["initial time step size", "time step variation"]
    default_values = {"time step variation": "fixed"}

    
class SolidMechanicsTimeParameters(_BaseTimeParameters):
    """
    Creates a SIERRA/SM Adagio parameters block.
    Users can add options to it using the 
    tools in :mod:`matcal.core.input_file_writer`.
    """ 

    type = "parameters for adagio region"
    required_keys = ["time increment"]
    default_values = {}


class TimeSteppingBlock(_BaseSierraInputFileBlock):
    type = "time stepping block"
    required_keys = ["start time"]
    default_values = {}

    def __init__(self, name, region_name, start_time, time_increment):
        """
        Creates a SIERRA/SM Adagio time stepping block.
        Users can add options to it using the 
        tools in :mod:`matcal.core.input_file_writer`.

        :param name: name for the time stepping period.
        :type name: str

        :param region_name: the name of the SIERRA/SM Adagio region that this applies to.
        :type region_name: str

        :param start_time: the start time for this time period.
        :type start_time: float

        :param time_increment: the desired time_increment for this time period.
        :type time_increment: float
        """ 
        super().__init__(name)
        self.set_start_time(start_time)
        self._time_parameters = SolidMechanicsTimeParameters(region_name, time_increment)
        self.add_subblock(self._time_parameters)

    def set_start_time(self, start_time):
        """
        Set or change the time block's start time. 

        :param start_time: the start time for this time period.
        :type start_time: float
        """
        self.add_line(InputFileLine("start time", start_time), replace=True)
    
    def set_time_increment(self, time_increment):
        """
        Set or change the time block's time increment. 

        :param time_increment: the time increment or time step for this time period.
        :type time_increment: float
        """
        self._time_parameters.set_time_increment(time_increment)


class TimeControl(_BaseSierraInputFileBlock):
    type = "time control"
    required_keys = ["termination time"]
    default_values = {}

    def __init__(self, termination_time, *time_stepping_blocks):
        """
        Creates a SIERRA time control input file block. 

        :param termination_time: the termination time for the simulation.
        :type termination_time: float

        :param time_stepping_blocks: the time stepping blocks that are desired 
            for defining time stepping periods for the simulation.
        :type time_stepping_blocks: list(:class:`matcal.sierra.input_file_writer.TimeSteppingBlock`)
        """
        super().__init__()
        self.set_termination_time(termination_time)
        for time_stepping_block in time_stepping_blocks:
            self.add_subblock(time_stepping_block)
        self._print_name=False
    
    def set_termination_time(self, termination_time):
        """
        Set or change the termination time for the time control block.

        :param termination_time: the desired termination time for the simulation.
        :type termination_time: float
        """
        self.add_line(InputFileLine("termination time", termination_time), replace=True)


class SolidMechanicsProcedure(_BaseSierraInputFileBlock):
    type = "adagio procedure"
    required_keys = []
    default_values = {}

    def __init__(self, name, solid_mechanics_region, start_time, termination_time, 
                 time_steps, init_time_step_scale_factor=1e-3):
        """
        Creates an SIERRA/SM Adagio procedure for a SIERRA input deck.

        :param name: The name of the Adagio procedure.
        :type name: str
        :param solid_mechanics_region: The region associated with the SIERRA/SM Adagio procedure.
        :type solid_mechanics_region: :class:`matcal.sierra.input_file_writer.SolidMechanicsRegion`
        :param start_time: The start time of the procedure.
        :type start_time: float
        :param termination_time: The termination time of the procedure.
        :type termination_time: float
        :param time_steps: The number of time steps for the procedure.
        :type time_steps: int
        :param init_time_step_scale_factor: Scale factor for the initial time step. Default is 1e-3.
        :type init_time_step_scale_factor: float
        """
        super().__init__(name)
        self._solid_mechanics_region = solid_mechanics_region
        self._start_time = start_time
        self._termination_time = termination_time
        self._time_steps = time_steps
        self._init_time_step_scale_factor = init_time_step_scale_factor
        self._time_step = None
        self._time_control_block = None
        self._set_time_step()
        self._set_small_time_step()
        self._init_elastic_time_step_block()
        self._init_load_time_stepping_block()
        self._add_time_control_block()

        self.add_subblock(solid_mechanics_region)

    def _set_time_step(self):
        self._time_step = (self._termination_time-self._start_time)/self._time_steps

    def _set_small_time_step(self):
        self._small_time_step = self._time_step*self._init_time_step_scale_factor

    def _init_elastic_time_step_block(self):
        self._elastic_time_step_block = TimeSteppingBlock("elastic_init", 
                                                          self._solid_mechanics_region.name, 
                                                          self._start_time, 
                                                          self._small_time_step)
    
    def _init_load_time_stepping_block(self):
        self._load_time_step_block = TimeSteppingBlock("load", self._solid_mechanics_region.name, 
                                                           self._start_time+self._small_time_step, 
                                                            self._time_step)

    def _add_time_control_block(self):
        self._time_control_block = TimeControl(self._termination_time, 
                                               self._elastic_time_step_block, 
                                               self._load_time_step_block)
        self.add_subblock(self._time_control_block)

    def _update_time_params(self):
        self._set_time_step()
        self._set_small_time_step()
        self._elastic_time_step_block.set_start_time(self._start_time)
        self._elastic_time_step_block.set_time_increment(self._small_time_step)
        self._load_time_step_block.set_start_time(self._start_time+self._small_time_step)
        self._load_time_step_block.set_time_increment(self._time_step)
        self._time_control_block.set_termination_time(self._termination_time)
       
    def set_number_of_time_steps(self, time_steps):
        """
        Set the number of time steps for the procedure.

        :param time_steps: The new number of time steps.
        :type time_steps: int
        """
        self._time_steps = time_steps
        self._update_time_params()

    def set_start_time(self, start_time):
        """
        Set the start time for the procedure.

        :param start_time: The new start time.
        :type start_time: float
        """
        self._start_time = start_time
        self._update_time_params()

    def set_end_time(self, end_time):
        """
        Set the termination time for the procedure.

        :param end_time: The new termination time.
        :type end_time: float
        """
        self._termination_time = end_time
        self._update_time_params()


class ArpeggioTransfer(_BaseSierraInputFileBlock):
    type = "transfer"
    required_keys = ["copy"]
    default_values = {}
    
    def __init__(self, name):
        """
        Creates a transfer operation input block for SIERRA.

        :param name: The name of the transfer operation.
        :type name: str
        """
        super().__init__(name)
    
    def _set_mesh_entity_copy(self, mesh_entity, sending_region, receiving_region):
        copy_line = InputFileLine("copy", "volume", mesh_entity, 
                                  "from", sending_region, "to", receiving_region)
        copy_line.suppress_symbol()
        self.add_line(copy_line, replace=True)

    def set_element_copy_transfer(self, sending_region, receiving_region):
        """
        Sets the transfer as a copy operation for elements between regions.

        :param sending_region: The region from which the element fields are copied.
        :type sending_region: str
        :param receiving_region: The region to which the element fields are copied.
        :type receiving_region: str
        """
        self._set_mesh_entity_copy("elements", sending_region, 
                                  receiving_region)
        
    def set_nodal_copy_transfer(self, sending_region, receiving_region):
        """
         Sets the transfer as a copy operation for nodes between regions.

        :param sending_region: The region from which the nodal fields are copied.
        :type sending_region: str
        :param receiving_region: The region to which the nodal fields are copied.
        :type receiving_region: str
        """
        self._set_mesh_entity_copy("nodes", sending_region, 
            receiving_region)
    
    def add_field_to_send(self, sending_field, receiving_field, sending_state="none", 
            receiving_state="none"):
        """
        Add a field to be transferred between regions.

        :param sending_field: The name of the field to send.
        :type sending_field: str
        :param receiving_field: The name of the field to receive.
        :type receiving_field: str
        :param sending_state: The state of the sending field (default is "none").
        :type sending_state: str
        :param receiving_state: The state of the receiving field (default is "none").
        :type receiving_state: str
        """
        name = self.get_line_name(sending_field, receiving_field, sending_state, receiving_state)
        send_line = InputFileLine("send", "field", sending_field, "state", sending_state,
                                  "to", receiving_field, "state", receiving_state, name=name)
        send_line.suppress_symbol()
        self.add_line(send_line)

    @staticmethod
    def get_line_name(sending_field, receiving_field, sending_state="none", 
            receiving_state="none"):
        return "_".join([sending_field, sending_state, receiving_field, receiving_state])

    def add_send_blocks(self, *blocks):
        send_blocks = " ".join(blocks)
        send_blocks_line = InputFileLine("send", "block", send_blocks, "to", 
                                         send_blocks, name="send_blocks")
        send_blocks_line.suppress_symbol()
        self.add_line(send_blocks_line)


class CoupledTransientParameters(_BaseSierraInputFileBlock):
    type = "parameters for transient"
    required_keys = ["start time", "termination time"]
    default_values = {}

    def __init__(self, name, thermal_region_name, solid_mechanics_region_name, 
            start_time, termination_time,time_step):
        """
        Creates a time stepping transient parameters block parameters 
        for a coupled SIERRA/SM - SIERRA/Thermal simulation.

        :param name: The name of the transient parameter block.
        :type name: str
        :param thermal_region_name: The name of the SIERRA/Thermal Aria region 
            associated with the transient parameters.
        :type thermal_region_name: str
        :param solid_mechanics_region_name: The name of the SIERRA/SM Adagio region associated with 
            the transient parameters.
        :type solid_mechanics_region_name: str
        :param start_time: The start time for the transient simulation.
        :type start_time: float
        :param termination_time: The termination time for the transient simulation.
        :type termination_time: float
        :param time_step: The time step size for the simulation.
        :type time_step: float
        """
        super().__init__(name)
        self._solid_mechanics_params = SolidMechanicsTimeParameters(solid_mechanics_region_name, time_step)
        self._thermal_params = ThermalTimeParameters(thermal_region_name, time_step)
        self.add_subblock(self._solid_mechanics_params)
        self.add_subblock(self._thermal_params)
        lines = {self.required_keys[0]:start_time, 
                 self.required_keys[1]:termination_time}
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
        else:
            raise ValueError("Thermal and solid mechanics time increments are not equal.")
    

class NonlinearStep(_BaseSierraInputFileBlock):
    type = "nonlinear"
    required_keys = []
    default_values = {}

    def __init__(self, name, *lines):
        """
        Creates an nonlinear solution step for use in SIERRA coupled simulations.

        :param name: The name of the nonlinear step.
        :type name: str
        :param lines: Optional input file lines to be added to the nonlinear step.
        :type lines: list(:class:`matcal.core.input_file_writer.InputFileLine`)
        """
        super().__init__(name)
        for line in lines:
            self.add_line(line)


class NonlinearParameters(_BaseSierraInputFileBlock):
    type = "parameters for nonlinear"
    required_keys = ["converged when"]
    default_values = {"converged when":
                      ('"thermal_region.MaxInitialNonlinearResidual(0) < 1.0e-8 '+
                       ' || CURRENT_STEP > 20"')}
    def __init__(self, *args, **kwargs):
        """
        Initialize the nonlinear step convergence criteria.
        Default is "thermal_region.MaxInitialNonlinearResidual(0) < 1.0e-8 || CURRENT_STEP > 20"

        :param args: Positional arguments passed to the base class.
        :type args: tuple
        :param kwargs: Keyword arguments passed to the base class.
        :type kwargs: dict
        """
        super().__init__(*args, **kwargs)
        self.set_symbol_for_lines(None)


class CoupledTransient(_BaseSierraInputFileBlock):
    type = "transient"
    required_keys = []
    default_values = {}

    def __init__(self, name, solid_region, thermal_region, nonlinear_step_name=None):
        """
        Create an Arpeggio/coupled SIERRA transient input file block. 
        If a nonlinear step name is provided, 
        it will create a loosely coupled Arpeggio transient solve.

        :param name: The name of the transient block.
        :type name: str
        :param solid_region: The name of the solid mechanics region to advance.
        :type solid_region: str
        :param thermal_region: The name of the thermal region to advance.
        :type thermal_region: str
        :param nonlinear_step_name: Optional name for the nonlinear step block. Default is None.
        :type nonlinear_step_name: str or None
        """
        super().__init__(name)
        self._nonlinear_step_name = nonlinear_step_name
        self._advance_solid = InputFileLine("advance", solid_region, name="advance_solid")
        self._advance_thermal = InputFileLine("advance", thermal_region, name="advance_thermal")
        self._post_solid_lines = []
        self._post_thermal_lines = []

    def _create_transfer_line(self, transfer_name):
        line_name = transfer_name
        transfer_line = InputFileLine("transfer", transfer_name, name=line_name)
        return transfer_line

    def add_transfer_post_solid_mechanics(self, transfer_name):
        """
        Add a transfer operation to be executed after solid mechanics.

        :param transfer_name: The name of the transfer operation.
        :type transfer_name: str
        """
        self._post_solid_lines.append(self._create_transfer_line(transfer_name))

    def add_transfer_post_thermal(self, transfer_name):
        """
        Add a transfer operation to be executed after the thermal solve.

        :param transfer_name: The name of the transfer operation.
        :type transfer_name: str
        """
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
            nonlinear_step_block = NonlinearStep(self._nonlinear_step_name, 
                                                      *self.lines.values())
            self._lines = OrderedDict()
            self.add_subblock(nonlinear_step_block, replace=True)

    def get_string(self):
        """
        Generate the string representation of the transient block.

        If a nonlinear step name is provided, wrap the transient block lines 
        in a NonlinearStep block.

        :return: The string representation of the transient block.
        :rtype: str
        """
        self._setup_lines()
       
        return super().get_string()


class CoupledSystem(_BaseSierraInputFileBlock):
    type = "system"
    required_keys = ['use initialize']
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
    required_keys = ['use system']
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

    def set_transient_time_parameters(self, transient_name, start_time, 
                                      end_time, time_step):
        transient_time_params = self.subblocks[transient_name]
        transient_time_params.set_start_time(start_time)
        transient_time_params.set_termination_time(end_time)
        transient_time_params.set_time_increment(time_step)


class Procedure(_BaseSierraInputFileBlock):
    type = "procedure"
    required_keys = []
    default_values = {}

    def __init__(self, solution_control_block, *transfers, 
                 name=get_default_coupled_procedure_name()):
        super().__init__(name)
        self._solution_control = solution_control_block
        self.add_subblock(solution_control_block)
        for transfer in transfers:
            self.add_subblock(transfer)


class FiniteElementParameters(_BaseSierraInputFileBlock):
    type = "parameters for block"
    required_keys = ["material"]
    default_values = {}

    def __init__(self, material_name, *blocks):
        name = ""
        for block in blocks:
            name += block + " "
        name = name.strip()
        super().__init__(name)
        material_line = InputFileLine(self.required_keys[0], material_name)
        self.add_line(material_line)


class SolidMechanicsFiniteElementParameters(FiniteElementParameters):
    type = "parameters for block"
    required_keys = (FiniteElementParameters.required_keys +
        ["model", "section"])
    default_values = {}

    def __init__(self, material_name, material_model, *blocks):
        super().__init__(material_name, *blocks)
        model_line = InputFileLine(self.required_keys[1], material_model)
        self.add_line(model_line)
        self.set_section(_SectionNames.total_lagrange)

    def set_section(self, section_name):
        section_line = InputFileLine(self.required_keys[2], section_name)
        self.add_line(section_line, replace=True)

    def get_section(self):
        return self.get_line_value(self.required_keys[2])

    def get_blocks(self):
        return list(self.name.split(" "))

class FiniteElementModelNames:
    solid_mechanics = "matcal_solid_mechanics_model"
    thermal = "matcal_thermal_model"


class FiniteElementModel(_BaseSierraInputFileBlock):
    type = "finite element model"
    required_keys = ["database name", "database type"]
    default_values = {required_keys[1]:"exodusII"}

    def __init__(self, *finite_element_model_parameters, 
                 name=FiniteElementModelNames.solid_mechanics):
        super().__init__(name)
        for finite_element_model_parameter in finite_element_model_parameters:
            self.add_subblock(finite_element_model_parameter)

    @property
    def mesh_filename(self):
        return self.lines[self.required_keys[0]].get_values()[-1]

    def set_mesh_filename(self, mesh_file):
        if self.required_keys[0] not in self.lines:
            self.add_line(InputFileLine(self.required_keys[0], mesh_file))
        else:
            self.lines[self.required_keys[0]].set(mesh_file, 1)

    def set_element_section(self, section_name):
        for block in self.subblocks.values():
            block.set_section(section_name)

    def get_element_section(self):
        sections = []
        for block in self.subblocks.values():
            sections.append(block.get_section())
        if len(set(sections)) == 1:
            return list(set(sections))[0]
        elif len(set(sections)) == 0:
            raise ValueError("No element sections found. Add FiniteElementModelParameters block " 
                "to this FiniteElementModel." )
        else:
            return set(sections)

    def get_blocks(self):
        blocks = []
        for subblock in self.get_subblocks_by_type(FiniteElementParameters.type):
            blocks += subblock.get_blocks()
        return blocks

class SolidMechanicsImplicitDynamics(_BaseSierraInputFileBlock):
    type = "implicit dynamics"
    required_keys = []
    default_values = {"contact timestep":"off"}
    def __init__(self):
        super().__init__()
        self.set_print_name(False)
        self.set_print_title()


class _BaseDeath(_BaseSierraInputFileBlock):
    type = "element death"
    required_keys = ["criterion"]
    
    def __init__(self, death_variable, critical_value, 
        criterion_eval_operator=">=", name="hades"):
        super().__init__(name=name)
        criterion_line = InputFileLine(self.required_keys[0], "element value of", death_variable, 
                                       criterion_eval_operator, critical_value)
        criterion_line.set_symbol("is")
        self.add_line(criterion_line)
    
    def get_critical_value(self):
        return self.get_line_value(self.required_keys[0], -1)


class SolidMechanicsDeath(_BaseDeath):
    type = "element death"
    required_keys = _BaseDeath.required_keys+["block"]
    default_values = {"skip criteria evaluation at start of load step":"on"}

    def __init__(self, death_variable, critical_value, *death_blocks, 
        criterion_eval_operator=">=", name="hades"):
        """
        Creates an element death block for SIERRA/SM

        :param death_variable: The variable used to evaluate the element death criterion.
        :type death_variable: str
        :param critical_value: The critical value of the death variable for element removal.
        :type critical_value: float
        :param death_blocks: One or more blocks where element death is applied.
        :type death_blocks: tuple[str]
        :param criterion_eval_operator: The operator used to evaluate 
            the criterion (e.g., ">=", "<="). Default is ">=".
        :type criterion_eval_operator: str
        :param name: The name of the element death block. Default is "hades".
        :type name: str
        """
        super().__init__(death_variable, critical_value, criterion_eval_operator, 
                         name=name)
        block_line = InputFileLine("block", *death_blocks)
        self.add_line(block_line)


class ThermalDeath(_BaseDeath):
    type = "element death"
    required_keys = _BaseDeath.required_keys+["Add volume"]
    default_values = {}

    def __init__(self, death_variable, critical_value, *death_blocks, 
        criterion_eval_operator=">=", name="hades"):
        """
        Creates an element death block for SIERRA/Thermal

        :param death_variable: The variable used to evaluate the element death criterion.
        :type death_variable: str
        :param critical_value: The critical value of the death variable for element removal.
        :type critical_value: float
        :param death_blocks: One or more blocks where element death is applied.
        :type death_blocks: tuple[str]
        :param criterion_eval_operator: The operator used to evaluate 
            the criterion (e.g., ">=", "<="). Default is ">=".
        :type criterion_eval_operator: str
        :param name: The name of the element death block. Default is "hades".
        :type name: str
        """
        super().__init__(death_variable, critical_value, criterion_eval_operator, 
                         name=name)
        volume_line = InputFileLine("Add volume", *death_blocks)
        volume_line.suppress_symbol()
        self.add_line(volume_line)


class _SolidMechanicsWithMeshEntity(_BaseSierraInputFileBlock):
    required_keys = []
    default_values = {}

    def __init__(self, mesh_entity_name=None,
                 mesh_entity="node set", **kwargs):
        super().__init__(**kwargs)
        if mesh_entity_name is not None:
            self._add_mesh_entity_line(mesh_entity_name=mesh_entity_name, 
                                       mesh_entity=mesh_entity)
    
    def _add_mesh_entity_line(self, mesh_entity_name, mesh_entity):
        if mesh_entity_name.lower().strip() == "include all blocks":
            mesh_entity_line = InputFileLine(mesh_entity_name)
        else:
            mesh_entity_line = InputFileLine(mesh_entity, mesh_entity_name)
        self.add_line(mesh_entity_line)
        self.set_print_name(False) 
        self.set_print_title()           

    def read_from_mesh(self, read_variable):
            read_variable_line = InputFileLine("read variable", read_variable)
            self.add_line(read_variable_line, replace=True)


class _SolidMechanicsBaseConditionWithFunction(_SolidMechanicsWithMeshEntity):

    def __init__(self, function_name=None,
                 scale_factor=1.0, **kwargs):
        super().__init__(**kwargs)
        if function_name is not None:
            function_line = InputFileLine("function", function_name)
            self.add_line(function_line)
        if scale_factor != 1:
            scale_factor_line = InputFileLine("scale factor", scale_factor)
            self.add_line(scale_factor_line)
    

class _SolidMechanicsBaseConditionWithDirection(_SolidMechanicsWithMeshEntity):

    def __init__(self, direction_name=None, direction_key="component", **kwargs):
        super().__init__(**kwargs)
        if direction_name is not None:
            direction_line = InputFileLine(direction_key, direction_name)
            self.add_line(direction_line)


class SolidMechanicsFixedDisplacement(_SolidMechanicsBaseConditionWithDirection):
    type = "fixed displacement"

    def __init__(self, mesh_entity_name, direction_name, 
                 mesh_entity="node set", direction_key="component"):
        name = mesh_entity_name+" "+direction_name
        super().__init__(mesh_entity_name=mesh_entity_name, name=name, 
                         mesh_entity=mesh_entity, direction_name=direction_name, 
                         direction_key=direction_key)
        

class SolidMechanicsPrescribedDisplacement(_SolidMechanicsBaseConditionWithFunction, 
                                                _SolidMechanicsBaseConditionWithDirection):
    type = "prescribed displacement"
    
    def __init__(self,  function_name,  mesh_entity_name, direction_name, 
                  mesh_entity="node set", direction_key="component", scale_factor=1.0):
        name = mesh_entity_name+" "+direction_name
        if function_name is not None:
            name +=" "+function_name
        super().__init__(function_name=function_name, mesh_entity_name=mesh_entity_name, 
                         direction_name=direction_name, 
                         mesh_entity=mesh_entity, direction_key=direction_key, 
                         scale_factor=scale_factor, name=name)


class SolidMechanicsPrescribedTemperature(_SolidMechanicsBaseConditionWithFunction):
    type = "prescribed temperature"
    
    def __init__(self,   mesh_entity_name,  
                 scale_factor=1.0, mesh_entity="node set", 
                 function_name=None, transfer=None):
        if function_name is not None:
            name = mesh_entity_name+" "+function_name
        elif transfer is not None:
            name = mesh_entity_name+" "+"temperature transfer"
        else:
            name = mesh_entity_name+" "+"read temperature from mesh"

        super().__init__(function_name=function_name, mesh_entity_name=mesh_entity_name, 
                         mesh_entity=mesh_entity, name=name, 
                         scale_factor=scale_factor)
        if transfer:
            transfer_line = InputFileLine("receive", "from", "transfer", name="transfer_temp")
            transfer_line.suppress_symbol()
            self.add_line(transfer_line)


class SolidMechanicsInitialTemperature(_SolidMechanicsWithMeshEntity):
    type = "initial temperature"

    def __init__(self, mesh_entity_name, magnitude, mesh_entity="block"):
        super().__init__(mesh_entity_name=mesh_entity_name, 
                        mesh_entity=mesh_entity)
        temp_mag_line = InputFileLine("magnitude", magnitude)
        self.add_line(temp_mag_line)
        

class SolidMechanicsUserOutput(_SolidMechanicsWithMeshEntity):
    type = "user output"
    required_keys = []
    default_values = {}
    
    def __init__(self, output_name, mesh_entity_name=None, mesh_entity=None,  
                 compute_interval="every step"):
        super().__init__(name=output_name, mesh_entity_name=mesh_entity_name, 
                         mesh_entity=mesh_entity)
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

    def add_compute_global_from_field(self, name, field, source_entity, 
                                      calculation="average"):
        self.add_compute_global(name, "as", calculation, "of", source_entity, 
                                field)

    def add_compute_global(self, name, *args):
        global_line = InputFileLine("compute", "global", name, *args, name="global "+name)
        global_line.suppress_symbol()
        self.add_line(global_line)

    def add_compute_element_from_element(self, name, source_name, 
                                         calculation="volume weighted average", replace=False):
        self.add_compute_element(name, "as", calculation, "of", "element", source_name)

    def add_compute_element(self, name, *args):
        element_line = InputFileLine("compute", "element", name, *args, name="element "+name)
        element_line.suppress_symbol()
        self.add_line(element_line, replace=True)

    def add_compute_element_as_function(self, name, function_name):
        self.add_compute_element(name, "as", "function", function_name)

    def add_nodal_variable_transformation(self, variable, transformed_name, 
        target_coordinate_system):

        transform_line = InputFileLine("transform nodal variable", variable, 
            "to coordinate system", target_coordinate_system, "as", transformed_name,
             name=transformed_name)
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
        var_type_line = InputFileLine("type", var_storage_location, var_type, 
                                      "length", "=", len(initial_values))
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
    required_keys = ["source variable", "target_variable", "radius", 
                     "distance algorithm"]
    
    def __init__(self, source_variable, target_variable, radius, 
                 distance_algorithm="euclidean_graph"):
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
    required_keys = ["source variable", "target_variable", "radius", 
                     "distance algorithm"]
    default_values = {}
    def __init__(self, radius, 
                 distance_algorithm="euclidean_graph"):
        super().__init__("damage_increment", "nonlocal_damage_increment", 
                         radius, distance_algorithm)


class _SolidMechanicsOutputStepIncrement(_BaseSierraInputFileBlock):
    def __init__(self, output_step_increment, **kwargs):
        super().__init__(**kwargs)
        self.set_output_step_increment(output_step_increment)

    def set_output_step_increment(self, output_step_increment):
        output_increment_line = InputFileLine("at step", 0, "," "increment", "=", 
                                             output_step_increment)
        output_increment_line.suppress_symbol()
        self.add_line(output_increment_line, replace=True)        


class _SolidMechanicsBaseOutput(_BaseSierraInputFileBlock):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.has_output_variables=False

    def add_output_variable(self, variable_scope, variable_name, save_as_name = None):
        new_line_args = [variable_scope, variable_name]
        if save_as_name is not None:
            check_value_is_nonempty_str(
                save_as_name, 
                "save_as_name", 
                "SolidMechanicsResultsOutputBlock.add_output_variable")
            new_line_args.append("as")
            new_line_args.append(save_as_name)
        name = self._get_line_name(variable_scope, variable_name, save_as_name)
        new_output_line = InputFileLine(*new_line_args, name=name)
        new_output_line.suppress_symbol()
        self.add_line(new_output_line)
        self.has_output_variables=True

    @staticmethod
    def _get_line_name(variable_scope, variable_name, save_as_name):
        name = variable_scope+" "+variable_name
        if save_as_name is not None:
            name += " "+save_as_name
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


class SolidMechanicsResultsOutput(_SolidMechanicsOutputStepIncrement,
                                       _SolidMechanicsBaseOutput):
    type = "results output"
    required_keys = ["database name", "database type", "at step"]
    default_values = {required_keys[1]:"exodusII"}
    nodal_header = "nodal"
    element_header = "element"

    def __init__(self, output_step_increment, results_output_file="./results/results.e", 
                 name="general_exodus_output"):
        """
        Create a exodus results output for a SIERRA/SM input file.

        :param output_step_increment: The step increment for results output.
        :type output_step_increment: int
        :param results_output_file: A filename with optional path in which to 
            write the results output. 
            Default is "./results/results.e".
        :type results_output_file: str
        """
        super().__init__(name=name, output_step_increment=output_step_increment)
        database_name_line = InputFileLine(self.required_keys[0], results_output_file)
        self.add_line(database_name_line)

    def add_nodal_output(self, variable_name, save_as_name=None):
        """
        Add a nodal output variable to the results output.

        :param variable_name: The name of the nodal variable to output.
        :type variable_name: str
        :param save_as_name: Optional name to save the variable as. Default is None.
        :type save_as_name: str or None
        """
        self.add_output_variable(self.nodal_header, variable_name, save_as_name)

    def add_element_output(self, variable_name, save_as_name=None):
        """
        Add an element output variable to the results output.

        :param variable_name: The name of the element variable to output.
        :type variable_name: str
        :param save_as_name: Optional name to save the variable as. Default is None.
        :type save_as_name: str or None
        """
        self.add_output_variable(self.element_header, variable_name, save_as_name)

    def add_include_surface(self, *surface_names):
        """
        Include specific surfaces in the results output.

        :param surface_names: Names of the surfaces to include.
        :type surface_names: tuple[str]
        """
        include_surf_line = InputFileLine("include", *surface_names)
        self.add_line(include_surf_line)
    
    def add_exclude_blocks(self, *blocks):
        """
        Exclude specific blocks from the results output.

        :param blocks: Names of the blocks to exclude.
        :type blocks: tuple[str]
        """
        exclude_blocks_line = InputFileLine("exclude", *blocks)
        self.add_line(exclude_blocks_line)

    def set_output_exposed_surface(self, output_exposed_surface=True):
        """
        Set whether to output just the exposed surface of the mesh.

        :param output_exposed_surface: If True, output just the exposed surface. Default is True.
        :type output_exposed_surface: bool
        """
        output_surface_key = "output mesh"
        if output_exposed_surface and output_surface_key not in self._lines:
            output_line = InputFileLine(output_surface_key, "exposed surface")
            self.add_line(output_line)
        elif not output_exposed_surface and output_surface_key in self._lines:
            self._lines.pop(output_surface_key)

    def _get_element_variable_line_name(self, element_variable_name, save_as_name=None):
        return _SolidMechanicsBaseOutput._get_line_name(self.element_header, 
                                                       element_variable_name, 
                                                       save_as_name)
    
    def _get_nodal_variable_line_name(self, nodal_variable_name, save_as_name=None):
        return _SolidMechanicsBaseOutput._get_line_name(self.nodal_header, 
                                                       nodal_variable_name, 
                                                       save_as_name)
    

class SolidMechanicsHeartbeatOutput(_SolidMechanicsOutputStepIncrement, 
                                         _SolidMechanicsBaseOutput):
    type = "heartbeat output"
    required_keys = ["stream name"]
    default_values = {"labels":"off", "legend":"on", "precision":16}

    def __init__(self, output_step_increment, *output_vars, filename="./results.csv"):
        """
        Creates a SIERRA/SM heartbeat output input block for text output in a 
        simulation. By default, this creates a CSV file with named headers.

        :param output_step_increment: The step increment for output generation.
        :type output_step_increment: int
        :param output_vars: Global output variables to include in the heartbeat output.
        :type output_vars: list(str)
        :param filename: The name of the output file. Default is "./results.csv".
        :type filename: str
        """
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
    default_values = {"terminate type":"entire_run"}
    
    def add_global_termination_criteria(self, global_variable, value, operator="<"):
        termination_line = InputFileLine("terminate", "global",
            global_variable, operator, value, name=f"global {global_variable}")
        termination_line.suppress_symbol()
        self.add_line(termination_line, replace=True)


class SolidMechanicsAdaptiveTimeStepping(_BaseSierraInputFileBlock):
    type = "adaptive time stepping"
    required_keys = ["minimum multiplier", "maximum multiplier"]
    default_values = {"maximum failure cutbacks":15}

    def __init__(self, minimum_multiplier=1e-8, maximum_multiplier=1):
        """
        Creates a SIERRA/SM adaptive time-stepping block.

        :param minimum_multiplier: The minimum multiplier for adaptive time-stepping. 
            Default is 1e-6.
        :type minimum_multiplier: float
        :param maximum_multiplier: The maximum multiplier for adaptive time-stepping. Default is 1.
        :type maximum_multiplier: float
        """
        super().__init__()
        self.set_print_name(False)
        self.set_minimum_multiplier(minimum_multiplier)
        self.set_maximum_multiplier(maximum_multiplier)
    
    def set_cutback_factor(self, cutback_factor=0.5):
        """
        Set the cutback factor for adaptive time-stepping.

        :param cutback_factor: The factor by which the time step is reduced during cutback.
            Default is 0.5.
        :type cutback_factor: float
        """
        cutback_factor_line = InputFileLine("cutback factor", cutback_factor)
        self.add_line(cutback_factor_line, replace=True)
    
    def set_growth_factor(self, growth_factor=1.5):
        """
        Set the growth factor for adaptive time-stepping.

        :param growth_factor: The factor by which the time step is increased during growth. 
            Default is 1.5.
        :type growth_factor: float
        """
        growth_factor_line = InputFileLine("growth factor", growth_factor)
        self.add_line(growth_factor_line, replace=True)

    def set_minimum_multiplier(self, minimum_multiplier=1e-6):
        """
        Set the minimum multiplier for adaptive time-stepping.

        :param minimum_multiplier: The minimum multiplier for the time step. Default is 1e-6.
        :type minimum_multiplier: float
        """
        min_mult_line = InputFileLine(self.required_keys[0], minimum_multiplier)
        self.add_line(min_mult_line, replace=True)
    
    def set_maximum_multiplier(self, maximum_multiplier=1):
        """
        Set the maximum multiplier for adaptive time-stepping.

        :param maximum_multiplier: The maximum multiplier for the time step. Default is 1.
        :type maximum_multiplier: float
        """
        max_mult_line = InputFileLine(self.required_keys[1], maximum_multiplier)
        self.add_line(max_mult_line, replace=True)

    def set_iteration_target(self, target_iterations=75, window=5):
        """
        Set the target number of iterations and the iteration window for adaptive time-stepping.

        :param target_iterations: The target number of iterations for convergence. Default is 75.
        :type target_iterations: int
        :param window: The size of the iteration window. Default is 5.
        :type window: int
        """
        target_iteration_line = InputFileLine("target iterations", 
                                target_iterations)
        self.add_line(target_iteration_line, replace=True)
        window_line = InputFileLine("iteration window", 
                                window)
        self.add_line(window_line, replace=True)
    
    def set_adaptive_time_stepping_method(self, method):
        """
        Set the method for adaptive time-stepping.

        :param method: The name of the adaptive time-stepping method to use.
        :type method: str
        """
        method_line = InputFileLine("method", method)
        self.add_line(method_line, replace=True)


class SolidMechanicsInteractionDefaults(_BaseSierraInputFileBlock):
    type = "interaction defaults"
    required_keys = ["friction model", "general contact", "self contact"]
    default_values = {"general contact": "on"}

    def __init__(self, friction_model_name, 
                       self_contact=True):
        super().__init__()
        self.set_print_name(False)
        self.set_self_contact(self_contact)
        friction_model_line = InputFileLine(self.required_keys[0], friction_model_name)
        self.add_line(friction_model_line)

    def set_self_contact(self, self_contact=True):
        contact_val="off"
        if self_contact:
            contact_val="on"
        self_contact_line = InputFileLine(self.required_keys[2], contact_val)
        self.add_line(self_contact_line, replace=True)


class SolidMechanicsConstantFrictionModel(_BaseSierraInputFileBlock):
    type = "constant friction model"
    required_keys = ["friction coefficient"]
    default_values = {}

    def __init__(self, friction_model_name, 
                       friction_coefficient=0.3):
        super().__init__(name=friction_model_name)
        self.set_friction_coefficient(friction_coefficient)

    def set_friction_coefficient(self, friction_coefficient):
        friction_coeff_line = InputFileLine(self.required_keys[0], friction_coefficient)
        self.add_line(friction_coeff_line, replace=True)

    def get_friction_coefficient(self):
        return self.get_line_value(self.required_keys[0])

class SolidMechanicsRemoveInitialOverlap(_BaseSierraInputFileBlock):
    type = "remove initial overlap"
    required_keys = []
    default_values = {}

    def __init__(self, ):
        super().__init__()
        self.set_print_name(False)
        

class SolidMechanicsContactDefinitions(_BaseSierraInputFileBlock):
    type = "contact definition"
    required_keys = []
    default_values = {"skin all blocks":"on"}

    def __init__(self, friction_model_block, name="contact_defs"):
        super().__init__(name=name)
        self.add_subblock(friction_model_block)
        interactions_default_block=SolidMechanicsInteractionDefaults(friction_model_block.name)
        self.add_subblock(interactions_default_block)
        self.add_subblock(SolidMechanicsRemoveInitialOverlap())

    def get_interaction_defaults_block(self):
        return self.subblocks[SolidMechanicsInteractionDefaults.type]
    
    def get_constant_friction_model_block(self):
        return self.get_subblock_by_type(SolidMechanicsConstantFrictionModel.type)
    
    def get_remove_initial_overlap_block(self):
        return self.subblocks[SolidMechanicsRemoveInitialOverlap.type]


class SolidMechanicsNonlinearSolverBase(_BaseSierraInputFileBlock):

    def __init__(self, name, target_relative_residual,
                 acceptable_relative_residual, 
                 minimum_iterations,  
                 maximum_iterations,
                 *args, **kwargs):
        super().__init__(name=name)
        self.set_minimum_iterations(minimum_iterations)
        self.set_acceptable_relative_residual(acceptable_relative_residual)
        self.set_target_relative_residual(target_relative_residual)        
        self.set_maximum_iterations(maximum_iterations)

    def set_minimum_iterations(self, minimum_iterations=5):
        min_iter_line = InputFileLine("minimum iterations", minimum_iterations)
        self.add_line(min_iter_line, replace=True)

    def set_acceptable_relative_residual(self, acceptable_relative_residual=1e-2):
        accept_resid_line = InputFileLine("acceptable relative residual", 
                                          acceptable_relative_residual)
        self.add_line(accept_resid_line, replace=True)

    def set_target_relative_residual(self, target_relative_residual=1e-2):
        target_resid_line = InputFileLine("target relative residual", 
                                          target_relative_residual)
        self.add_line(target_resid_line, replace=True)

    def set_target_residual(self, target_residual=1e-2):
        target_resid_line = InputFileLine("target residual", 
                                          target_residual)
        self.add_line(target_resid_line, replace=True)

    def set_acceptable_residual(self, acceptable_residual=1e-2):
        accept_resid_line = InputFileLine("acceptable residual", 
                                          acceptable_residual)
        self.add_line(accept_resid_line, replace=True)

    def set_maximum_iterations(self, maximum_iterations=100):
        max_iters = InputFileLine("maximum iterations", 
                                   maximum_iterations)
        self.add_line(max_iters, replace=True)

    def get_target_relative_residual(self):
        return self.get_line_value("target relative residual")

    def get_target_residual(self):
        return self.get_line_value("target residual")

    def get_acceptable_relative_residual(self):
        return self.get_line_value("acceptable relative residual")

    def get_acceptable_residual(self):
        if "acceptable residual" in self.lines:
            return self.get_line_value("acceptable residual")
        else:
            return None


class SolidMechanicsControlContact(SolidMechanicsNonlinearSolverBase):
    type = "control contact"
    required_keys = []
    default_values = {"lagrange adaptive penalty":"off",
                      "lagrange initialize":"none", 
                      "lagrange maximum updates":0}

    def __init__(self, target_relative_residual=1e-3, 
                 acceptable_relative_residual=1e-2, minimum_iterations=5,
                 maximum_iterations=100,      
                 name="contact_control"):
        super().__init__(name=name, 
                         target_relative_residual=target_relative_residual, 
                         acceptable_relative_residual=acceptable_relative_residual, 
                         minimum_iterations=minimum_iterations, 
                         maximum_iterations=maximum_iterations)


class SolidMechanicsLoadstepPredictor(_BaseSierraInputFileBlock):
    type = "loadstep predictor"
    required_keys = ["scale factor"]
    default_values = {}

    def __init__(self, scale_factor=0.0,):
        super().__init__()
        self.set_print_name(False)
        self.set_scale_factor(scale_factor)

    def set_scale_factor(self, scale_factor=0.0):
        min_iter_line = InputFileLine("scale factor", scale_factor)
        self.add_line(min_iter_line, replace=True)


class SolidMechanicsFullTangentPreconditioner(_BaseSierraInputFileBlock):
    type = "full tangent preconditioner"
    required_keys = []
    default_values = {"small number of iterations":20, 
                      "minimum smoothing iterations":15, 
                      "iteration update":25 }

    def __init__(self, linear_solver=None):
        """
        Create a full tangent preconditioner block for a SIERRA/SM input file.

        :param linear_solver: Optional name of the linear solver to use with this preconditioner.
            This linear solve must be defined in the input deck elsewhere.
            Default is None.
        :type linear_solver: str
        """
        super().__init__()
        self.set_print_name(False)
        self.set_linear_solver(linear_solver)
        
    def set_linear_solver(self, linear_solver=None):
        """
        Set the linear solver for the full tangent preconditioner.

        :param linear_solver: Optional name of the linear solver to use with this preconditioner.
            This linear solve must be defined in the input deck elsewhere.
            Default is None.
        :type linear_solver: str
        """
        if linear_solver is not None:
            linear_solver_line = InputFileLine("linear solver", linear_solver.name)
            self.add_line(linear_solver_line, replace=True)


class SolidMechanicsConjugateGradient(SolidMechanicsNonlinearSolverBase):
    type = "cg"
    required_keys = []
    default_values = {"reference":"Belytschko"}

    def __init__(self, target_relative_residual=1e-9, 
                 acceptable_relative_residual=1e-8, 
                 minimum_iterations=15, maximum_iterations=100, 
                 full_tangent_preconditioner=None,
                 name=None):
        """
        Create a conjugate gradient solver input block for SIERRA?SM .

        :param target_relative_residual: The target relative residual for 
            convergence. Default is 1e-9.
        :type target_relative_residual: float
        :param acceptable_relative_residual: The acceptable relative residual for 
            convergence. Default is 1e-8.
        :type acceptable_relative_residual: float
        :param minimum_iterations: The minimum number of iterations for the solver. Default is 15.
        :type minimum_iterations: int
        :param maximum_iterations: The maximum number of iterations for the solver. Default is 100.
        :type maximum_iterations: int
        :param full_tangent_preconditioner: Optional full tangent preconditioner block. 
            Default is None.
        :type full_tangent_preconditioner: None or
            :class:`matcal.sierra.input_file_writer.SolidMechanicsFullTangentPreconditioner`
        :param name: Optional name for the solver block. Default is None.
        :type name: str or None
        """
        super().__init__(name=name, 
                         target_relative_residual=target_relative_residual, 
                         acceptable_relative_residual=acceptable_relative_residual, 
                         minimum_iterations=minimum_iterations, 
                         maximum_iterations=maximum_iterations)        
        self.set_print_name(False)
        self.set_full_tangent_preconditioner(full_tangent_preconditioner)
        self.set_target_residual(target_relative_residual*100)

    def set_full_tangent_preconditioner(self, full_tangent_preconditioner=None):
        """
        Set or remove the full tangent preconditioner block.

        :param full_tangent_preconditioner: The full tangent preconditioner block to add. 
        If None and a preconditioner is pre-existing, the preconditioner is removed.
        :type full_tangent_preconditioner: None or
            :class:`matcal.sierra.input_file_writer.SolidMechanicsFullTangentPreconditioner`
        """
        if full_tangent_preconditioner is not None:
            self.add_subblock(full_tangent_preconditioner)
        elif SolidMechanicsFullTangentPreconditioner.type in self.subblocks:
            self.remove_subblock(SolidMechanicsFullTangentPreconditioner.type)


class SolidMechanicsNonlinearSolverContainer(_BaseSierraInputFileBlock):
    type = "solver"
    required_keys = []
    default_values = {}


class SolidMechanicsRegion(_BaseSierraInputFileBlock):
    type = "adagio region"
    required_keys = ["use finite element model"]
    default_values = {}

    def __init__(self, name, finite_element_model_name):
        super().__init__(name=name)
        finite_element_model_line = InputFileLine(self.required_keys[0], 
                                                  finite_element_model_name)
        finite_element_model_line.suppress_symbol()
        self.add_line(finite_element_model_line)
        

class _Coupling:
    adiabatic = "adiabatic"
    staggered = "staggered"
    iterative = "iterative"


class _Failure:
    local_failure = "local"
    nonlocal_failure = "nonlocal"


class SierraFileBase(_BaseTypedInputFileBlock):
    required_keys = []
    default_values = {}
    type = "sierra"
    _load_bc_function_name = "prescribed_displacement"
    _temperature_bc_function_name = "prescribed_temperature"
    _material_model_line_name = "material_file_line"

    def __init__(self, material, death_blocks, **kwargs):
        super().__init__("matcal_generated_SIERRA_model", begin_end=True,
                         **kwargs)
        self._death_blocks = death_blocks
        self._add_material_file_to_input(material)
        self._sm_finite_element_model = FiniteElementModel()
        self.add_subblock(self._sm_finite_element_model)

        global_defs = SierraGlobalDefinitions()
        self.add_lines(*global_defs.lines.values())
        self.add_subblock(FetiSolver())
        self.add_subblock(GdswSolver())
        self.add_subblock(TotalLagrangeSection())
        self.add_subblock(SolidSectionDefault())
        ct_section = TotalLagrangeSection()
        ct_section.use_composite_tet()
        self.add_subblock(ct_section)
        
        self._adaptive_time_stepping = None
        self._cg = None
        self._solver = None
        self._full_tangent_preconditioner = None
        self._exodus_output = None
        self._default_element_output = ["hydrostatic_stress", "von_mises", "log_strain"]
        self._default_nodal_output = ["displacement"]
        self._vol_average_user_output = SolidMechanicsUserOutput("vol_average", "include all blocks")
        self._coupling = None
        self._failure = None
        self._death = None
        self._initial_temp = None
        self._solution_termination = None

        self._solid_mechanics_region = self._build_default_solid_mechanics_region()
        sm_procedure_name = get_default_solid_mechanics_procedure_name()
        self._solid_mechanics_procedure = SolidMechanicsProcedure(sm_procedure_name, 
            self._solid_mechanics_region, 0, 1, 300)
        self.add_subblock(self._solid_mechanics_procedure)
        self._heartbeat_output = None
        self._reset_heartbeat_output()

        self._start_time_user_supplied = False
        self._end_time_user_supplied = False  

    def _clear_default_element_output_field_names(self):
        self._default_element_output = []

    def _reset_heartbeat_output(self):
        self._heartbeat_output = SolidMechanicsHeartbeatOutput(1, "time")
        self._solid_mechanics_region.add_subblock(self._heartbeat_output, replace=True)

    @property
    def cg(self):
        """
        Returns the conjugate gradient solver block for the input file.

        :rtype: :class:`matcal.sierra.input_file_writer.SolidMechanicsConjugateGradient`
        """
        return self._cg

    @property
    def adative_time_stepping(self):
        """
        Returns the adaptive time stepping block for the input file.

        :rtype: :class:`matcal.sierra.input_file_writer.SolidMechanicsAdaptiveTimeStepping`
        """
        return self._adaptive_time_stepping

    @property
    def full_tangent_preconditioner(self):
        """
        Returns the full tangent preconditioner block for the input file.

        :rtype: :class:`matcal.sierra.input_file_writer.SolidMechanicsFullTangentPreconditioner`
        """
        return self._full_tangent_preconditioner

    @property
    def solid_mechanics_region(self):
        """
        Returns the SIERRA/SM Adagio region input block for the input file.

        :rtype: :class:`matcal.sierra.input_file_writer.SolidMechanicsRegion`
        """
        return self._solid_mechanics_region

    @property
    def solid_mechanics_procedure(self):
        """
        Returns the SIERRA/SM Adagio prcedure input block for the input file.

        :rtype: :class:`matcal.sierra.input_file_writer.SolidMechanicsProcedure`
        """
        return self._solid_mechanics_procedure

    @property
    def death(self):
        """
        Returns the SIERRA/SM death input block for the input file.

        :rtype: :class:`matcal.sierra.input_file_writer.SolidMechanicsDeath`
        """
        return self._death
    
    @property
    def solid_mechanics_finite_element_model(self):
        """
        Returns the SIERRA/SM finite element model block from the input file.

        :rtype: :class:`matcal.sierra.input_file_writer.FiniteElementModel`
        """
        return self._sm_finite_element_model

    @property
    def solid_mechanics_element_section(self):
        """
        Returns the SIERRA/SM finite element section block from the input file.

        :rtype: :class:`matcal.sierra.input_file_writer.SolidSectionDefault` or
            :class:`matcal.sierra.input_file_writer.TotalLagrangeSection`
        """
        return self._get_section_subblock()

    @property
    def exodus_output(self):
        """
        Returns the SIERRA/SM exodus results output block from the input file.

        :rtype: :class:`matcal.sierra.input_file_writer.SolidMechanicsResultsOutput` 
        """
        return self._exodus_output
    
    @property
    def heartbeat_output(self):
        """
        Returns the SIERRA/SM heatbeat results output block from the input file.

        :rtype: :class:`matcal.sierra.input_file_writer.SolidMechanicsHeartbeatOutput` 
        """
        return self._heartbeat_output

    @property
    def element_type(self):
        """Returns the element type being used for the model.
    
        :rtype: str"""
        return self._sm_finite_element_model.get_element_section()

    @property
    def prescribed_loading_boundary_condition(self):
        """
        Returns the input block for the SIERRA function 
        describing the prescribed loading for the model.

        :rtype: :class:`~matcal.sierra.input_file_writer.PiecewiseLinearFunction`
            or :class:`~matcal.sierra.input_file_writer.AnalyticSierraFunction`
        """
        if self._load_bc_function_name in self.subblocks:
            return self.subblocks[self._load_bc_function_name]
        else:
            return None

    @property
    def prescribed_temperature_boundary_condition(self):
        """
        Returns the input block for the SIERRA function 
        describing the prescribed temperature for the model.

        :rtype: :class:`~matcal.sierra.input_file_writer.PiecewiseLinearFunction`
            or :class:`~matcal.sierra.input_file_writer.AnalyticSierraFunction`
        """
        if self._temperature_bc_function_name in self.subblocks:
            return self.subblocks[self._temperature_bc_function_name]
        else:
            return None

    @property
    def initial_temperature(self):
        """
        Returns the input block for the SIERRA model intial temperature.

        :rtype: :class:`~matcal.sierra.input_file_writer.SolidMechanicsInitialTemperature`
        """
        return self._initial_temp

    @property
    def solution_termination(self):
        """
        Returns the input block for the SIERRA model solution termination.

        :rtype: :class:`~matcal.sierra.input_file_writer.SolidMechanicsSolutionTermination`
        """
        return self._solution_termination

    def _add_material_file_to_input(self, material):
        material_str = get_string_from_text_file(material.filename)
        self._material_file_line = InputFileLine(material_str, 
            name=self._material_model_line_name)
        self._material_file_line.suppress_symbol()
        self.add_line(self._material_file_line)

    def _build_default_solid_mechanics_region(self):
        region_name = get_default_solid_mechanics_region_name()
        
        region = SolidMechanicsRegion(region_name, self._sm_finite_element_model.name)
        self._adaptive_time_stepping = SolidMechanicsAdaptiveTimeStepping(1e-8, 1)
        self._adaptive_time_stepping.set_iteration_target()
        region.add_subblock(self._adaptive_time_stepping)
        self._solution_termination = SolidMechanicsSolutionTermination()
        region.add_subblock(self._solution_termination)
        gdsw = GdswSolver()
        self._full_tangent_preconditioner = SolidMechanicsFullTangentPreconditioner(gdsw)
        self._cg = SolidMechanicsConjugateGradient(full_tangent_preconditioner=
                                                       self._full_tangent_preconditioner)
        self._solver = SolidMechanicsNonlinearSolverContainer()
        self._solver.add_subblock(self._cg)
        region.add_subblock(self._solver)
        return region

    def _reset_state_boundary_conditions_and_output(self):
        self._reset_state_displacement_conditions()
        self._reset_state_temperature_conditions()
        sm_region = self.solid_mechanics_region
        sm_region.remove_subblocks_by_type(SolidMechanicsUserOutput.type)
        self._reset_heartbeat_output()
        self._solid_mechanics_region.add_subblock(self._vol_average_user_output)

    def _get_section_subblock(self):
        section_name = self._sm_finite_element_model.get_element_section()
        if section_name in self.subblocks:
            return self.get_subblock(section_name)
        else:
            return None
    
    def _activate_exodus_output(self, output_step_interval=20):
        if self._exodus_output is None:
            self._exodus_output = SolidMechanicsResultsOutput(output_step_interval)
            self._solid_mechanics_region.add_subblock(self._exodus_output)
            for ele_output in self._default_element_output:
                self._add_element_output_variable(ele_output)
            for nodal_output in self._default_nodal_output:
                self._add_nodal_output_variable(nodal_output)
        else:
            self._exodus_output.set_output_step_increment(output_step_interval)

    def _add_element_output_variable(self, *element_variable_names, 
                                    volume_average=True):
        if self._exodus_output is None:
            self._activate_exodus_output()
        for element_variable_name in element_variable_names:
            check_value_is_nonempty_str(element_variable_name, "element_variable_name", 
                                             "SierraModel.add_element_output_variable")
            save_name = None
            if volume_average:
                element_variable_name, save_name = self._add_volume_averaged_element_output(element_variable_name)
            self._add_element_output(element_variable_name, save_name)

    def _add_volume_averaged_element_output(self, element_variable_name):
        save_name = element_variable_name
        element_variable_name = self._get_volume_averaged_name(element_variable_name)
        self._vol_average_user_output.add_compute_element_from_element(element_variable_name, 
                                                                           save_name, replace=True)
        return element_variable_name, save_name

    def _add_element_output(self, element_variable_name, save_name):
        if save_name is not None:
            self._remove_element_mesh_output(save_name)
        else:
            to_remove = self._get_volume_averaged_name(element_variable_name)
            self._remove_element_mesh_output(to_remove, element_variable_name)
        if not self._element_variable_in_mesh_output(element_variable_name, save_name):
            self._exodus_output.add_element_output(element_variable_name, save_name)
        else:
            self._warn_user_element_variable_already_being_output(element_variable_name)

    def _remove_element_mesh_output(self, element_variable_name, save_name=None):
        if self._element_variable_in_mesh_output(element_variable_name, save_name):
            line_name = self._exodus_output._get_element_variable_line_name(element_variable_name, 
                                                                           save_name)
            self._exodus_output._lines.pop(line_name)

    def _get_volume_averaged_name(self, element_variable_name):
        return element_variable_name+"_vol_avg"

    def _element_variable_in_mesh_output(self, element_variable_name, save_as_name=None):
        if self._exodus_output is not None:
            line_name = self._exodus_output._get_element_variable_line_name(element_variable_name, 
                                                                           save_as_name)
            return line_name in self._exodus_output.lines
        else:
            return False

    def _warn_user_element_variable_already_being_output(self, element_variable_name):
        warn_str = (f"Element variable '{element_variable_name}' "+
                    "already output. Not adding it to output again.")
        logger.warning(warn_str)

    def _add_nodal_output_variable(self, *nodal_variable_names):
        if self._exodus_output is None:
            self._activate_exodus_output()
        for nodal_variable_name in nodal_variable_names:
            check_value_is_nonempty_str(nodal_variable_name, "nodal_variable_name", 
                                             "SierraModel.add_nodal_output_variable")
            if not self._nodal_variable_in_mesh_output(nodal_variable_name):
                self._exodus_output.add_nodal_output(nodal_variable_name)
            else:
                self._warn_user_nodal_variable_already_being_output(nodal_variable_name)
    
    def _warn_user_nodal_variable_already_being_output(self, nodal_variable_name):
        warn_str = (f"Nodal variable '{nodal_variable_name}' "
                    +"already output. Not adding it to output again.")
        logger.warning(warn_str)

    def _nodal_variable_in_mesh_output(self, nodal_variable_name):
        if self._exodus_output is not None:
            line_name = self._exodus_output._get_nodal_variable_line_name(nodal_variable_name)
            return line_name in self._exodus_output.lines
        else:
            return False

    def _add_new_default_mesh_output(self, variable_name, nodal=False):
        if self.exodus_output_active:
            if not nodal:
                self._add_element_output_variable(variable_name) 
            else:
                self._add_nodal_output_variable(variable_name)
        elif variable_name not in self._default_nodal_output and nodal:
            self._default_nodal_output.append(variable_name)
        elif variable_name not in self._default_element_output and not nodal:
            self._default_element_output.append(variable_name)

    def _activate_adiabatic_heating(self):
        self._coupling = _Coupling.adiabatic
        self._add_temperature_output()

    def _add_temperature_output(self, nodal=False):
        self._add_new_default_mesh_output(TEMPERATURE_KEY, nodal)

    def _use_under_integrated_element(self):
        self._sm_finite_element_model.set_element_section(_SectionNames.uniform_gradient)

    def _use_total_lagrange_element(self):
        self._sm_finite_element_model.set_element_section(_SectionNames.total_lagrange)

    def _add_solid_mechanics_finite_element_parameters(self, material_name, material_model,
                                                      *blocks):
        new_finite_element_parameters = SolidMechanicsFiniteElementParameters(material_name, 
            material_model, *blocks)
        self._sm_finite_element_model.add_subblock(new_finite_element_parameters)

    def _set_start_time(self, start_time):
        self._start_time_user_supplied = True
        self._update_time_parameters(start_time=start_time)

    def _set_end_time(self, end_time):
        self._end_time_user_supplied = True
        self._update_time_parameters(termination_time=end_time)

    def _set_number_of_time_steps(self, number_of_steps):
        self._update_time_parameters(number_of_steps=number_of_steps)

    def _update_time_parameters(self, start_time=None, termination_time=None, 
                                number_of_steps=None):
        if start_time is not None:
            self._solid_mechanics_procedure.set_start_time(start_time)
        if termination_time is not None:
            self._solid_mechanics_procedure.set_end_time(termination_time)
        if number_of_steps is not None:
            self._solid_mechanics_procedure.set_number_of_time_steps(number_of_steps)

    @property
    def exodus_output_active(self):
        return self._exodus_output is not None and self._exodus_output.has_output_variables

    def _set_state_prescribed_temperature_from_boundary_data(self, boundary_data, 
                                                             state, temperature_key):
        full_field_temp_data = self._get_temperature_boundary_data_full_field(boundary_data, state, 
            temperature_key)
        if full_field_temp_data is not None:
            self._set_temperature_from_mesh_full_field_data(full_field_temp_data, temperature_key)
        elif full_field_temp_data is None:
            self._set_temperature_from_temperature_time_history(boundary_data, state, 
                temperature_key)

    def _get_temperature_boundary_data_full_field(self, boundary_data, state, 
                                                  temperature_key):
        full_field_temp_data = None
        data_list = boundary_data[state]
        for data in data_list:
            if len(data[temperature_key].shape) == 2:
                full_field_temp_data = data
                if len(data_list) > 1:
                    logger.warning(f"There are more than one data sets "+
                                   f"supplied as boundary data for the model {self.name} "+
                                   f"for state {data.state.name} and at least one "+
                                   " is full field data. Only "+
                                   "the first full field data set will be used.")
                self._add_temperature_output(nodal=True)
                break
        return full_field_temp_data

    def _set_temperature_from_mesh_full_field_data(self, full_field_temp_data, 
                                                   temperature_key):
        prescribed_temp = SolidMechanicsPrescribedTemperature("include all blocks")
        prescribed_temp.read_from_mesh(temperature_key)
        self._solid_mechanics_region.add_subblock(prescribed_temp)

    def _set_temperature_from_temperature_time_history(self, boundary_data, state, 
                                                       temperature_key):
        temperature_function = get_temperature_function_from_data_collection(boundary_data, 
                                                                                state, 
                                                                                temperature_key=
                                                                                temperature_key)
        func_name = self._temperature_bc_function_name
        prescribed_temp = SolidMechanicsPrescribedTemperature("include all blocks", 
                                                                function_name=func_name)
        self._solid_mechanics_region.add_subblock(prescribed_temp)
        temp_function = PiecewiseLinearFunction(func_name, temperature_function[TIME_KEY], 
                                                temperature_function[temperature_key])
        self.add_subblock(temp_function)
        self._add_temperature_output(nodal=True)

    def _set_initial_temperature_from_parameters(self, model_parameters_by_precedent):
        if TEMPERATURE_KEY in model_parameters_by_precedent.keys():
            initial_temp_value = model_parameters_by_precedent[TEMPERATURE_KEY]
            self._initial_temp = SolidMechanicsInitialTemperature("include all blocks", 
                                                            initial_temp_value)
            self._solid_mechanics_region.add_subblock(self._initial_temp, replace=True)
            self._add_temperature_output()

        elif self._coupling != None:
            raise RuntimeError(f"When running a coupled simulation, "
                                "a model constant, state parameter or "+
                                "study parameter named \"temperature\" is "
                                f"required for all states. Check input for model {self.name}")

    def _add_prescribed_loading_boundary_condition_with_displacement_function(self,
            displacement_function, node_sets, directions, direction_keys, scale_factor):
        self._set_time_parameters_to_loading_function(displacement_function, scale_factor)
        func = PiecewiseLinearFunction(self._load_bc_function_name, displacement_function[TIME_KEY], 
                                       displacement_function[DISPLACEMENT_KEY])
        func.scale_function(x=scale_factor, y=scale_factor)     
        self.add_subblock(func)
        self._add_prescribed_displacement_boundary_condition(self._load_bc_function_name, 
        node_sets, directions, direction_keys)
        
    def _add_prescribed_displacement_boundary_condition(self, function_name, 
            node_sets, directions, direction_keys, read_variables=None):
        if read_variables is None:
            read_variables = [None]*len(node_sets)
        bc_inputs = zip(node_sets, directions, direction_keys, read_variables)
        for node_set, direction, direction_key, read_variable in bc_inputs:
            prescribed_disp = SolidMechanicsPrescribedDisplacement(function_name, 
                                node_set, direction, direction_key=direction_key)
            if read_variable is not None:
                prescribed_disp.read_from_mesh(read_variable)
            self._solid_mechanics_region.add_subblock(prescribed_disp)

    def _set_time_parameters_to_loading_function(self, bc_function_array, scale_factor):
        if not self._start_time_user_supplied:
            self._update_time_parameters(start_time=min(bc_function_array[TIME_KEY])*scale_factor)
        if not self._end_time_user_supplied:
            end_time = max(bc_function_array[TIME_KEY])*scale_factor
            self._update_time_parameters(termination_time=end_time)

    def _set_local_mesh_filename(self, mesh_name):
        self._sm_finite_element_model.set_mesh_filename(mesh_name)

    def _set_fixed_boundary_conditions(self, node_sets, directions):
        for node_set, direction in zip(node_sets, directions):
            fixed_disp = SolidMechanicsFixedDisplacement(node_set, direction)
            self._solid_mechanics_region.add_subblock(fixed_disp)
       
    def _reset_state_temperature_conditions(self):
        if self._temperature_bc_function_name in self.subblocks:
            self.remove_subblock(self._temperature_bc_function_name)
        prescribed_temp_type = SolidMechanicsPrescribedTemperature.type
        self._solid_mechanics_region.remove_subblocks_by_type(prescribed_temp_type)
        self._solid_mechanics_region.remove_subblocks_by_type(SolidMechanicsInitialTemperature.type)
        
    def _reset_state_displacement_conditions(self):
        if self._load_bc_function_name in self.subblocks:
            self.remove_subblock(self._load_bc_function_name)
        prescribed_disp_type = SolidMechanicsPrescribedDisplacement.type
        self._solid_mechanics_region.remove_subblocks_by_type(prescribed_disp_type)

    def _add_heartbeat_global_variable(self, global_var, save_as_name=None):
        if not self._heartbeat_output.has_global_output(global_var, save_as_name):
            self._heartbeat_output.add_global_output(global_var, save_as_name)
        else:
            logger.warning(f"Global variable \'{global_var}\' already added. "+
                        "Not adding it again.")
            
    def _activate_element_death(self, death_variable="damage", critical_value=0.15):
        self._failure = _Failure.local_failure
        self._death = SolidMechanicsDeath(death_variable, critical_value, 
                                          *self._death_blocks)
        self._solid_mechanics_region.add_subblock(self._death, replace=True)
        self._add_new_default_mesh_output(death_variable)
        self._add_new_default_mesh_output("death_status")

    @property
    def coupling(self):
        return self._coupling

    @property
    def failure(self):
        return self._failure

    def _set_convergence_tolerance(self, nonlinear_solver, 
        target_relative_residual, target_residual=None, 
        acceptable_relative_residual=None, acceptable_residual=None, 
        target_resid_factor=100, acceptable_relative_resid_factor=10):
        nonlinear_solver.set_target_relative_residual(target_relative_residual)
        if target_residual is None:
            nonlinear_solver.set_target_residual(target_relative_residual*target_resid_factor)
        else:
            nonlinear_solver.set_target_residual(target_residual)

        if acceptable_relative_residual is None:
            nonlinear_solver.set_acceptable_relative_residual(target_relative_residual*
                acceptable_relative_resid_factor)
        else:
            nonlinear_solver.set_acceptable_relative_residual(acceptable_relative_residual)

        if acceptable_residual is not None:
            nonlinear_solver.set_acceptable_residual(acceptable_residual)

    def _set_cg_convergence_tolerance(self,  target_relative_residual, target_residual=None, 
        acceptable_relative_residual=None, acceptable_residual=None):
        self._set_convergence_tolerance(self._cg, target_relative_residual, target_residual, 
            acceptable_relative_residual, acceptable_residual)


class ThermalRegion(_BaseSierraInputFileBlock):
    type = "aria region"
    required_keys = ["use finite element model", "use linear solver"]
    default_values = {}
    def __init__(self, name, finite_element_model_name, solver):
        super().__init__(name=name)
        finite_element_model_line = InputFileLine(self.required_keys[0], 
                                                  finite_element_model_name)
        finite_element_model_line.suppress_symbol()
        self.add_line(finite_element_model_line, replace=True)
        self.add_solver(solver)
        self.add_nonlinear_solve_options()
        self.add_equations_to_solve(aria_quadrature_rule="Q1")
        
    def add_solver(self, solver):
        solver_line = InputFileLine(self.required_keys[1], solver.name)
        solver_line.suppress_symbol()
        self.add_line(solver_line, replace=True)

    def add_nonlinear_solve_options(self):
        strategy_line = InputFileLine("nonlinear solution strategy", "NEWTON")
        self.add_line(strategy_line, replace=True)
        max_iter_line = InputFileLine("maximum nonlinear iterations", 250)
        self.add_line(max_iter_line, replace=True)
        residual_tol_line = InputFileLine("nonlinear residual tolerance", 1e-8)
        self.add_line(residual_tol_line, replace=True)
        relax_factor_line = InputFileLine("nonlinear relaxation factor", 1.0)
        self.add_line(relax_factor_line, replace=True)

    def add_equations_to_solve(self, aria_quadrature_rule):
        energy_eq = InputFileLine("EQ energy for TEMPERATURE on all_blocks using",
                                  aria_quadrature_rule, "with DIFF SRC MASS")
        energy_eq.suppress_symbol()
        self.add_line(energy_eq, replace=True)
        mesh_eq = InputFileLine("EQ mesh for MESH_DISPLACEMENTS on all_blocks using",
                                  aria_quadrature_rule, "with XFER")
        mesh_eq.suppress_symbol()
        self.add_line(mesh_eq, replace=True)
    
    def add_heating_source(self, plastic_work_variable, num_integration_pts):
        self.remove_subblocks_by_type(SolidMechanicsUserVariable.type)
        plastic_work_var = SolidMechanicsUserVariable(plastic_work_variable, "element", "real",
                                    *[0]*num_integration_pts)
        blocks_line = InputFileLine("Add part", "all_blocks")
        blocks_line.suppress_symbol()
        plastic_work_var.add_line(blocks_line)
        self.add_subblock(plastic_work_var, replace=True)
        source_line = InputFileLine("source for energy on all_blocks",  
                                    "user_field_volume_heating name =", plastic_work_variable, 
                                    "scaling = 1")
        self.add_line(source_line, replace=True)

    def add_element_death(self, *death_blocks):
        death_status_aria_line = InputFileLine("User field real element scalar "+
            "death_status_aria on all_blocks value", 1)
        self.add_line(death_status_aria_line, replace=True)
        transfer_line = InputFileLine("transfer", "element death")
        transfer_line.suppress_symbol()
        self.add_line(transfer_line)
        thermal_death = ThermalDeath("death_status_aria", 0.99, *death_blocks, 
            criterion_eval_operator="<=")
        self.add_subblock(thermal_death, replace=True)

    def add_initial_condition(self, initial_temperature):
        initial_temperature_line = InputFileLine("IC const on all_blocks TEMPERATURE", 
                                                 initial_temperature)
        self.add_line(initial_temperature_line, replace=True)
        
    def add_dirichlet_temperature_boundary_condition(self, nodeset_name, temperature):
        bc_line = InputFileLine(f"BC Const Dirichlet at {nodeset_name} Temperature", temperature)
        self.add_line(bc_line, replace=True)


class SierraFileWithCoupling(SierraFileBase):

    def __init__(self, material, death_blocks, **kwargs):
        super().__init__(material, death_blocks, **kwargs)
        self._thermal_bc_nodesets = None
        self._coupling_transfers = []
        self._coupled_procedure = None
        self._thermal_model = None
        self._thermal_region = None
        self._thermal_material = None
        self._transient_params_1 = None
        self._transient_params_2 = None
        self._transient1 = None
        self._transient2 = None
        self._thermal_material = None
        self._death_transfer = None
        self._plastic_work_variable = None
        self._work_transfer = None

    def _set_thermal_bc_nodesets(self, nodesets):
        self._thermal_bc_nodesets=nodesets

    def _add_heating_source_element_variables(self, plastic_work_variable):
        if self._thermal_region is not None:
            self._reset_plastic_work_transfer_lines(plastic_work_variable)
            if self.element_type is _SectionNames.composite_tet:
                res  = self._add_volume_averaged_element_output(plastic_work_variable)
                vol_averaged_name, save_name = res
                num_integration_points = 1
                self._work_transfer.add_field_to_send(vol_averaged_name, plastic_work_variable)
                self._thermal_region.add_equations_to_solve("Q2")
            else:
                num_integration_points = self._get_num_integration_points()             
                self._work_transfer.add_field_to_send(plastic_work_variable, 
                                                      plastic_work_variable, 'new')
                self._thermal_region.add_equations_to_solve("Q1")
            self._thermal_region.add_heating_source(plastic_work_variable, num_integration_points)

    def _reset_plastic_work_transfer_lines(self, plastic_work_variable):
            for line_name in self._work_transfer.lines:
                if plastic_work_variable in line_name:
                    self._work_transfer.lines.pop(line_name)


    def _activate_thermal_coupling(self, thermal_conductivity, 
                                    density, specific_heat, 
                                    plastic_work_variable):
        self._plastic_work_variable = plastic_work_variable
        self._thermal_material = ThermalMaterial(density, thermal_conductivity, 
                                           specific_heat)
        self.add_subblock(self._thermal_material)
        thermal_solver = TpetraSolver()
        self.add_subblock(thermal_solver)
        self._thermal_model = FiniteElementModel(name=FiniteElementModelNames.thermal)
        self._add_sm_model_mesh_name_line_to_thermal_model()
        self._add_thermal_finite_element_parameters(*self._sm_finite_element_model.get_blocks())
        region_name = get_default_thermal_region_name()
        self._thermal_region = ThermalRegion(region_name, self._thermal_model.name, 
                                          thermal_solver)
        arpeggio_init=self._get_arpeggio_initialize()
        transients = self._get_arpeggio_transients(plastic_work_variable)        
        
        arpeggio_sys = CoupledSystem("main", arpeggio_init.name, *transients)

        transient_params = self._get_arpeggio_transient_parameters()
        sltn_ctl = SolutionControl("coupling", arpeggio_sys, arpeggio_init, 
                                   *transient_params)
        self._coupled_procedure = Procedure(sltn_ctl, *self._coupling_transfers)
        self._add_heating_source_element_variables(plastic_work_variable)
        self.remove_subblock(self._solid_mechanics_procedure)
        self.add_subblock(self._coupled_procedure)
        self._coupled_procedure.add_subblock(self._solid_mechanics_region)
        self._coupled_procedure.add_subblock(self._thermal_region)
        self._coupling = _Coupling.staggered
        self._add_temperature_output(nodal=True)

    def _add_sm_model_mesh_name_line_to_thermal_model(self):
        mesh_name_line_name = FiniteElementModel.required_keys[0]
        sm_mesh_name_line = self._sm_finite_element_model.lines[mesh_name_line_name]
        self._thermal_model.add_line(sm_mesh_name_line) 
        self.add_subblock(self._thermal_model)

    def _get_arpeggio_initialize(self):
        arpeggio_init = CoupledInitialize("initialization", self._solid_mechanics_region.name, 
                                                  self._thermal_region.name)
        disp_transfer_init = ArpeggioTransfer("solid_mechanics_to_thermal_disps_initial")
        disp_transfer_init.set_nodal_copy_transfer(self._solid_mechanics_region.name, 
            self._thermal_region.name)
        disp_transfer_init.add_field_to_send("DISPLACEMENT", "solution->MESH_DISPLACEMENTS", 
                                        "NEW", "OLD")
        disp_transfer_init.add_field_to_send("DISPLACEMENT", "solution->MESH_DISPLACEMENTS", 
                                        "NEW", "NEW")
        self._coupling_transfers.append(disp_transfer_init)
        
        temp_transfer_init = ArpeggioTransfer("thermal_to_solid_mechanics_initial")
        temp_transfer_init.set_nodal_copy_transfer(self._thermal_region.name, 
                                                   self._solid_mechanics_region.name)
        temp_transfer_init.add_field_to_send("solution->TEMPERATURE", "TEMPERATURE", 
                                             "NEW", "OLD")
        temp_transfer_init.add_field_to_send("solution->TEMPERATURE", "TEMPERATURE", 
                                             "NEW", "NEW")
        self._coupling_transfers.append(temp_transfer_init)

        arpeggio_init.add_transfer_post_solid_mechanics(disp_transfer_init.name)
        arpeggio_init.add_transfer_post_thermal(temp_transfer_init.name)
        return arpeggio_init
    
    def _add_post_solid_mechanics_transfer_to_transients(self, transfer):
        if transfer not in self._transient1.lines:
            self._transient1.add_transfer_post_solid_mechanics(transfer)
            self._transient1._setup_lines()
        
        if transfer not in self._transient2.lines:
            self._transient2.add_transfer_post_solid_mechanics(transfer)
            self._transient2._setup_lines()

    def _add_post_thermal_transfer_to_transients(self, transfer):
        if transfer not in self._transient1.lines:
            self._transient1.add_transfer_post_thermal(transfer)
        if transfer not in self._transient2.lines:
            self._transient2.add_transfer_post_thermal(transfer)
        self._transient1._setup_lines()
        self._transient2._setup_lines()
        
    def _get_arpeggio_transients(self, plastic_work_variable_name):
        self._transient1 = CoupledTransient("transient1", self._solid_mechanics_region.name, 
            self._thermal_region.name)
        self._transient2 = CoupledTransient("transient2", self._solid_mechanics_region.name, 
            self._thermal_region.name)
        self._add_transfers_to_transients(plastic_work_variable_name)

        return self._transient1, self._transient2
    
    def _add_transfers_to_transients(self, plastic_work_variable_name):
        transfers = self._get_transient_transfers(plastic_work_variable_name)
        disp_transfer, work_transfer, temp_transfer, death_transfer = transfers

        self._add_post_solid_mechanics_transfer_to_transients(disp_transfer.name)
        self._add_post_solid_mechanics_transfer_to_transients(work_transfer.name)
        self._add_post_thermal_transfer_to_transients(temp_transfer.name)

        if death_transfer is not None:
            self._add_post_solid_mechanics_transfer_to_transients(death_transfer.name)

    def _get_transient_transfers(self, plastic_work_variable_name):
        disp_transfer = ArpeggioTransfer("solid_mechanics_to_thermal_disps")
        disp_transfer.set_nodal_copy_transfer(self._solid_mechanics_region.name, 
            self._thermal_region.name)
        disp_transfer.add_field_to_send("DISPLACEMENT", "solution->MESH_DISPLACEMENTS", 
                                        "NEW", "New")
        self._coupling_transfers.append(disp_transfer)

        self._work_transfer = ArpeggioTransfer("solid_mechanics_to_thermal_work")
        self._work_transfer.set_element_copy_transfer(self._solid_mechanics_region.name, 
            self._thermal_region.name)
        self._work_transfer.add_field_to_send(plastic_work_variable_name, 
                                        plastic_work_variable_name, "new")
        self._coupling_transfers.append(self._work_transfer)

        temp_transfer = ArpeggioTransfer("thermal_to_solid_mechanics")
        temp_transfer.set_nodal_copy_transfer(self._thermal_region.name, 
                                                   self._solid_mechanics_region.name)
        temp_transfer.add_field_to_send("solution->TEMPERATURE", "TEMPERATURE", 
                                             "NEW", "OLD")
        self._coupling_transfers.append(temp_transfer)
        self._set_death_transfer()
        return disp_transfer, self._work_transfer, temp_transfer, self._death_transfer

    def _set_death_transfer(self):
        if self._death is not None and self._death_transfer is None:
            self._death_transfer = ArpeggioTransfer("solid_mechanics_to_thermal_death_status")
            self._death_transfer.set_element_copy_transfer(self._solid_mechanics_region.name, 
                                                   self._thermal_region.name)
            self._death_transfer.add_field_to_send("death_status", "death_status_aria", 
                                             "NEW")
            self._death_transfer.add_send_blocks(*self._death_blocks)
            self._coupling_transfers.append(self._death_transfer)
            self._thermal_region.add_element_death(*self._death_blocks)  
        return self._death_transfer

    def _activate_element_death(self, *args, **kwargs):
        super()._activate_element_death(*args, **kwargs)
        if self._thermal_region is not None:
            self._set_death_transfer()
            self._coupled_procedure.add_subblock(self._death_transfer, replace=True)
            self._add_post_solid_mechanics_transfer_to_transients(self._death_transfer.name)

    def _get_arpeggio_transient_parameters(self):
        time_params = self._get_solid_mechanics_time_params()
        start_time, termination_time, init_time_step, time_step = time_params
        self._transient_params_1 = CoupledTransientParameters(self._transient1.name, 
                                                         self._thermal_region.name, 
                                                         self._solid_mechanics_region.name,
                                                         start_time, 
                                                         start_time+init_time_step, 
                                                         init_time_step)
        self._transient_params_2 = CoupledTransientParameters(self._transient2.name, 
                                                         self._thermal_region.name,
                                                         self._solid_mechanics_region.name, 
                                                         start_time+init_time_step, 
                                                         termination_time, 
                                                         time_step)
        return self._transient_params_1, self._transient_params_2
        
    def _get_solid_mechanics_time_params(self):
        start_time = self._solid_mechanics_procedure._start_time
        termination_time = self._solid_mechanics_procedure._termination_time
        init_time_step_scale_factor = self._solid_mechanics_procedure._init_time_step_scale_factor
        time_step = self._solid_mechanics_procedure._time_step
        init_time_step = time_step*init_time_step_scale_factor
        return start_time, termination_time, init_time_step, time_step

    @property
    def is_coupled_simulation(self):
        return self._coupling is not None and self._coupling != _Coupling.adiabatic

    def _update_time_parameters(self, start_time=None, termination_time=None, number_of_steps=None):
        super()._update_time_parameters(start_time=start_time, 
                                        termination_time=termination_time, 
                                        number_of_steps=number_of_steps)
        if self.is_coupled_simulation:
            time_params = self._get_solid_mechanics_time_params()
            start_time, termination_time, init_time_step, time_step = time_params
            sltn_ctrl = self._coupled_procedure._solution_control
            sltn_ctrl.set_transient_time_parameters(self._transient1.name, 
                start_time, start_time+init_time_step, init_time_step)
            sltn_ctrl.set_transient_time_parameters(self._transient2.name, 
                start_time+init_time_step, termination_time, time_step)

    def _add_thermal_finite_element_parameters(self, *blocks):
        new_finite_element_parameters = FiniteElementParameters(self._thermal_material.name,
            *blocks)
        self._thermal_model.add_subblock(new_finite_element_parameters)

    def _set_initial_temperature_from_parameters(self, model_parameters_by_precedent):
        super()._set_initial_temperature_from_parameters(model_parameters_by_precedent)
        if self.is_coupled_simulation:
            temperature = model_parameters_by_precedent[TEMPERATURE_KEY]
            self._set_dirichlet_temperature_boundary_conditions(self._thermal_bc_nodesets, 
                temperature)
            self._thermal_region.add_initial_condition(temperature)

    def _set_dirichlet_temperature_boundary_conditions(self, node_sets, magnitude):
        for node_set in node_sets:
            self._thermal_region.add_dirichlet_temperature_boundary_condition(node_set, magnitude)

    def _activate_iterative_coupling(self):
        self._coupling = _Coupling.iterative
        nonlinear_step1_params = NonlinearParameters(name="converge_step_1")
        self._transient1.set_nonlinear_step_name(nonlinear_step1_params.name)
        self._coupled_procedure._solution_control.add_subblock(nonlinear_step1_params)
        nonlinear_step2_params = NonlinearParameters(name="converge_step_2")
        self._transient2.set_nonlinear_step_name(nonlinear_step2_params.name)
        self._coupled_procedure._solution_control.add_subblock(nonlinear_step2_params)

    def _use_total_lagrange_element(self, composite_tet=False):
        if composite_tet:
            self._sm_finite_element_model.set_element_section(_SectionNames.composite_tet)
        else:
            super()._use_total_lagrange_element()
        self._add_heating_source_element_variables(self._plastic_work_variable)

    def _use_under_integrated_element(self):
        super()._use_under_integrated_element()
        self._add_heating_source_element_variables(self._plastic_work_variable)

    def _get_num_integration_points(self):
        integration_pts = 8
        if self.element_type == _SectionNames.composite_tet:
            integration_pts = 4
        elif self.element_type == _SectionNames.uniform_gradient:
            integration_pts = 1
        return integration_pts


class SierraFileThreeDimensional(SierraFileWithCoupling):
    def __init__(self, material, death_blocks, **kwargs):
        super().__init__(material, death_blocks, **kwargs)
        self._implicit_dynamics = None
        self._solution_termination_output = None
        self._setup_solution_termination()
        self._full_field_output = None
        self._nonlocal_functions = []
        self._damage_increment_user_output = None
        self._nonlocal_average_output = None
        self._nonlocal_damage_increment_user_output = None
        self._nonlocal_damage_user_variables = []
    
    def _setup_solution_termination(self):
        self._solution_termination.add_global_termination_criteria("terminate_solution", 0.5, ">")

    def _add_solution_termination_user_output(self, termination_variable, max_var_drop):
        max_var_name = f"max_{termination_variable}"
        max_user_var = SolidMechanicsUserVariable(max_var_name, "global", "real", 0.0)
        max_user_var.add_global_operator()
        self._solid_mechanics_region.add_subblock(max_user_var, replace=True)
        self._add_max_termination_user_output(termination_variable, max_var_drop, 
                                      max_var_name)

    def _add_max_termination_user_output(self, termination_variable, max_var_drop, 
        max_var_name):
        self._solution_termination_output = SolidMechanicsUserOutput("solution_termination")
        self._solution_termination_output.add_compute_global_from_expression(max_var_name, 
            f"({termination_variable} >= {max_var_name}) ? {termination_variable} "+
            f": {max_var_name};")                                               
        end_time = self._solid_mechanics_procedure._termination_time
        start_time = self._solid_mechanics_procedure._start_time
        one_tenth_run_time = (end_time-start_time)/10+start_time
        self._solution_termination_output.add_compute_global_from_expression("terminate_solution", 
            f"(time >= {one_tenth_run_time}) ? "+
             f"({termination_variable} < {max_var_name}*(1-{max_var_drop})) ? 1 : 0"+
            f": 0;")     
        self._solid_mechanics_region.add_subblock(self._solution_termination_output)

    def _activate_full_field_results_output(self, results_file, *model_blocks):
        self._full_field_output = SolidMechanicsResultsOutput(1, results_file, 
            name="full_field_output")
        self._full_field_output.add_include_surface("full_field_data_surface")
        self._full_field_output.add_exclude_blocks(*model_blocks)
        global_outputs = self._heartbeat_output.get_global_outputs()
        for global_output in global_outputs:
            self._full_field_output.add_line(global_output)
        if self.coupling == _Coupling.adiabatic:
            self._full_field_output.add_element_output("temperature")
        elif self.coupling is not None:
            self._full_field_output.add_nodal_output("temperature")            
        self._full_field_output.add_nodal_output("displacement")
        self._full_field_output.set_output_exposed_surface()
        self._solid_mechanics_region.add_subblock(self._full_field_output, replace=True)

    def _add_nonlocal_user_output(self, death_variable, nonlocal_radius):
        self._reset_nonlocal_input()
        self._failure = _Failure.nonlocal_failure
        num_integration_pts = self._get_num_integration_points()
        self._add_nonlocal_damage_user_variables(num_integration_pts)
        self._add_damage_increment_user_output(death_variable, num_integration_pts)
        self._add_nonlocal_average_output(nonlocal_radius)
        self._add_nonlocal_damage_increment_user_output(death_variable, num_integration_pts)

    def _reset_nonlocal_input(self):
        for func in self._nonlocal_functions:
            self.remove_subblock(func)
        self._nonlocal_functions = []

        sm_subblocks = self._solid_mechanics_region.subblocks
        for user_var in self._nonlocal_damage_user_variables:
            if user_var.name in sm_subblocks:
                self._solid_mechanics_region.remove_subblock(user_var)
        self._nonlocal_damage_user_variables = []

        if self._damage_increment_user_output is not None:
            if self._damage_increment_user_output.name in sm_subblocks:
                self._solid_mechanics_region.remove_subblock(self._damage_increment_user_output)
                self._damage_increment_user_output = None
        if self._nonlocal_average_output is not None:
            if self._nonlocal_average_output.name in sm_subblocks:
                self._solid_mechanics_region.remove_subblock(self._nonlocal_average_output)
                self._nonlocal_average_output = None
        if self._nonlocal_damage_increment_user_output is not None:
            if self._nonlocal_damage_increment_user_output.name in sm_subblocks:
                self._solid_mechanics_region.remove_subblock(self._nonlocal_damage_increment_user_output)
                self._nonlocal_damage_increment_user_output = None
    
    def _add_nonlocal_damage_user_variables(self, num_integration_pts):
        initial_values = [0]*num_integration_pts
        damage_increment_var = SolidMechanicsUserVariable("damage_increment", "element",
                                                          "real", *initial_values)
        damage_increment_var.add_blocks(*self._death_blocks)
        self._nonlocal_damage_user_variables.append(damage_increment_var)
        self._solid_mechanics_region.add_subblock(damage_increment_var)
       
        nl_damage_increment_var = SolidMechanicsUserVariable("nonlocal_damage_increment", "element",
                                                          "real", *initial_values)
        nl_damage_increment_var.add_blocks(*self._death_blocks)
        self._nonlocal_damage_user_variables.append(nl_damage_increment_var)
        self._solid_mechanics_region.add_subblock(nl_damage_increment_var)

    def _get_death_blocks_string(self):
        return " ".join(self._death_blocks)

    def _add_damage_increment_user_output(self, death_variable, num_integration_pts):
        damage_inc_output = SolidMechanicsUserOutput("damage_increment_output", 
            self._get_death_blocks_string(), "block")
        for int_pt in range(num_integration_pts):
            int_pt +=1
            func_name = f"get_damage_increment_{int_pt}"
            damage_inc_output.add_compute_element_as_function(f"damage_increment({int_pt})", 
                                                              func_name)
            func = AnalyticSierraFunction(func_name)
            func.add_expression_variable("d_cur", "element", f"{death_variable}({int_pt})", 
                                         "state", "new")
            func.add_expression_variable("d_old", "element", f"{death_variable}({int_pt})", 
                                         "state", "old")
            func.add_evaluation_expression("d_cur - d_old")
            self._nonlocal_functions.append(func)
            self.add_subblock(func)
        self._damage_increment_user_output = damage_inc_output
        self._solid_mechanics_region.add_subblock(damage_inc_output)

    def _add_nonlocal_average_output(self, nonlocal_radius):
        nl_damage_avg_output = SolidMechanicsUserOutput("nonlocal_damage_average",
            self._get_death_blocks_string(), "block")
        nonlocal_avg = SolidMechanicsNonlocalDamageAverage(nonlocal_radius)
        nl_damage_avg_output.add_subblock(nonlocal_avg)
        self._nonlocal_average_output = nl_damage_avg_output
        self._solid_mechanics_region.add_subblock(nl_damage_avg_output)

    def _add_nonlocal_damage_increment_user_output(self, death_variable, num_integration_pts):
        critical_value = self._death.get_critical_value()
        expression = (f"(d_old + nl_damage_inc) < {critical_value} ? d_old + nl_damage_inc :"+
                f"{critical_value};")
        nl_damage_inc_output = SolidMechanicsUserOutput("nonlocal_damage_increment_output",
            self._get_death_blocks_string(), "block")
        for int_pt in range(num_integration_pts):
            int_pt+=1
            func_name = f"apply_nonlocal_damage_increment_{int_pt}"
            nl_damage_inc_output.add_compute_element_as_function(f"damage({int_pt})", 
                                                              func_name)
            func = AnalyticSierraFunction(func_name)
            func.add_expression_variable("nl_damage_inc", "element", 
                                         f"nonlocal_damage_increment({int_pt})", 
                                         "state", "new")
            func.add_expression_variable("d_old", "element", f"{death_variable}({int_pt})", 
                                         "state", "old")
            func.add_evaluation_expression(expression)
            self._nonlocal_functions.append(func)
            self.add_subblock(func)
        self._nonlocal_damage_increment_user_output = nl_damage_inc_output
        self._solid_mechanics_region.add_subblock(nl_damage_inc_output)

    def _activate_implicit_dynamics(self):
        self._implicit_dynamics = SolidMechanicsImplicitDynamics()
        self._solid_mechanics_region.add_subblock(self._implicit_dynamics)


class SierraFileThreeDimensionalContact(SierraFileThreeDimensional):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._contact_definitions = None
        self._friction_model = None
        self._control_contact = None
        self._contact_target_relative_residual = 1e-3
        self._contact_target_residual = 1e-2
        self._contact_acceptable_relative_residual = 1e-2
        self._contact_acceptable_residual = None
    
    def _self_contact(self):
        return self._contact_definitions is not None
        
    def _activate_self_contact(self, friction_coefficient):

        if not self._self_contact():
            self._friction_model = SolidMechanicsConstantFrictionModel('self_contact_friction', 
                friction_coefficient)
            self._contact_definitions = SolidMechanicsContactDefinitions(self._friction_model)
            self._solid_mechanics_region.add_subblock(self._contact_definitions)
            self._setup_contact_solver()
        else:
            logger.warning(f"Self contact already active. Updating friction coefficient "+
                           "to {friction_coefficient}.")
            self._friction_model.set_friction_coefficient(friction_coefficient)

    def _setup_contact_solver(self):
        self._add_control_contact_block()
        self._load_step_predictor = SolidMechanicsLoadstepPredictor()
        self._solver.add_subblock(self._load_step_predictor)

    def _add_control_contact_block(self):
        self._control_contact = SolidMechanicsControlContact()
        self._set_contact_convergence_tolerance(self._contact_target_relative_residual, 
            self._contact_target_residual, self._contact_acceptable_relative_residual, 
            self._contact_acceptable_residual)
        self._solver.add_subblock(self._control_contact)

    def _set_contact_convergence_tolerance(self, target_relative_residual, target_residual=None, 
        acceptable_relative_residual=None, acceptable_residual=None):
        self._contact_target_relative_residual = target_relative_residual
        if target_residual is not None:
            self._contact_target_residual = target_residual
        if acceptable_relative_residual is not None:
            self._contact_acceptable_relative_residual = acceptable_relative_residual
        if acceptable_residual  is not None:
            self._contact_acceptable_residual = acceptable_residual
        if self._control_contact is not None:
            self._set_convergence_tolerance(self._control_contact, target_relative_residual, 
                target_residual,acceptable_relative_residual, acceptable_residual, 
                target_resid_factor=10)
            self._update_cg_solver_convergence_tolerances()

    def _update_cg_solver_convergence_tolerances(self):
        contact_target_resid = self._control_contact.get_target_relative_residual()
        super()._set_cg_convergence_tolerance(target_relative_residual=contact_target_resid/10, 
                                        target_residual=contact_target_resid*10, 
                                        acceptable_relative_residual=10)

    