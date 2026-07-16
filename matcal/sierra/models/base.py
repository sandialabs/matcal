"""
Base and shared classes for MatCal SIERRA models.

This module contains:
- SierraModelBase: common simulation execution helpers (check-input, check-syntax, extra args)
- UserDefinedSierraModel: wrapper for user-supplied SIERRA input + mesh + aux files
- Base classes for MatCal-generated SIERRA models (standard, coupled, 3D, contact, gripped-specimen family)

Notes:
- Concrete, user-facing models (tension, shear/torsion, VFM, material point) live in other modules.
- Base classes are internal (prefixed with "_") and are not intended to be re-exported publicly.
"""

from abc import abstractmethod
from collections import OrderedDict
import numbers
import os

from matcal.core.boundary_condition_calculators import (    
    format_bc_function_comment_lines,
    get_displacement_function_from_load_displacement_data_collection,
)
from matcal.core.constants import (
    DISPLACEMENT_KEY,
    LOAD_KEY,
    TEMPERATURE_KEY,
    TIME_KEY,
)
from matcal.core.data import Data, DataCollection
from matcal.core.logger import initialize_matcal_logger
from matcal.core.models import (
    AdditionalFileCopyPreprocessor,
    InputFileCopyPreprocessor,
    ModelBase,
)
from matcal.core.parameters import (
    _get_parameters_according_to_precedence,
    _get_parameters_source_according_to_precedence,
)
from matcal.core.utilities import (
    matcal_name_format,
    check_value_is_nonempty_str,
    check_value_is_real_between_values,
    check_item_is_correct_type,
    check_value_is_positive_integer,
    check_value_is_positive_real,
    _convert_list_of_files_to_abs_path_list,
    check_value_is_nonnegative_real
)

from matcal.full_field.data_importer import FieldSeriesData

from matcal.sierra.material import Material
from matcal.sierra.input_file_writer import (
    SierraFileBase,
    SierraFileWithCoupling,
    SierraFileThreeDimensional,
    SierraFileThreeDimensionalContact,
    SolidMechanicsUserOutput,
    _Coupling,
)
from matcal.sierra.simulators import SierraSimulator

from .preprocessors import AddApreproParamFileLinesPreprocessor, DecomposeAndCopyMeshPreprocessor

logger = initialize_matcal_logger(__name__)


class SierraModelBase(ModelBase):
    """
    Common base class for running SIERRA executables through MatCal.
    """

    _simulator_class = SierraSimulator

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._additional_executable_arguments = []
        self._check_syntax = False
        self._check_input = False

    def set_time_limit(self, time_limit_hours):

        if time_limit_hours > 96:
            message = (
                "Sierra model time limit cannot exceed 96 hours.\n"
                f"Requested time limit was {time_limit_hours} hours."
            )
            raise ValueError(message)
        super().set_time_limit(time_limit_hours)

    def _get_simulator_class_inputs(self, state):
        args = [
            self.name,
            self._simulation_information,
            self._results_information,
            state,
            self._input_filename,
        ]
        kwargs = {
            "custom_commands": self._additional_executable_arguments,
            "check_syntax": self._check_syntax,
            "check_input": self._check_input,
            "epu_results": self._epu_results(),
            "model_constants": self.get_model_constants(state),
        }
        return args, kwargs

    def _epu_results(self):
        exodus_reader = (self._results_information.results_reader_object == FieldSeriesData)
        parallel = self._simulation_information.number_of_cores > 1
        return bool(exodus_reader and parallel)

    def add_executable_argument(self, argument):
        """
        Pass an additional argument directly to the SIERRA executable.
        """
        if not isinstance(argument, str):
            message = (
                "Sierra arguments need to be passed as strings.\n"
                + f" Flag Passed: {argument}\nFlag Type: {type(argument)}"
            )
            raise TypeError(message)
        self._additional_executable_arguments.append(argument)

    def run_check_input(self, state, parameter_collection, target_directory=None):
        """
        Run with SIERRA '--check-input'.
        """
        self._check_input = True
        results = super().run(state, parameter_collection, target_directory)
        self._check_input = False
        return results

    def run_check_syntax(self, state, parameter_collection, target_directory=None):
        """
        Run with SIERRA '--check-syntax'.
        """
        self._check_syntax = True
        results = super().run(state, parameter_collection, target_directory)
        self._check_syntax = False
        return results


class UserDefinedSierraModel(SierraModelBase):
    """
    Use a user-provided SIERRA input deck + mesh + optional include files/dirs.
    """

    model_type = "user_defined_sierra_model"

    def __init__(self, executable, simulation_input_file, simulation_mesh_filename, *other_sources):
        super().__init__(executable=executable)
        self._input_filename = os.path.abspath(simulation_input_file)
        self._mesh_filename = os.path.abspath(simulation_mesh_filename)
        self._additional_sources_to_copy = _convert_list_of_files_to_abs_path_list(other_sources)

    def _setup_state(self, state, state_template_dir=".", build_mesh=True):
        ifile_copier = InputFileCopyPreprocessor()
        ifile_copier.process(state_template_dir, input_filename=self._input_filename)

        aprepro_preprocessor = AddApreproParamFileLinesPreprocessor()
        aprepro_preprocessor.process(state_template_dir, input_filename=self._input_filename)

        additional_file_copier = AdditionalFileCopyPreprocessor()
        additional_file_copier.process(state_template_dir, self._additional_sources_to_copy)

        if build_mesh:
            mesh_decomposer = DecomposeAndCopyMeshPreprocessor()
            mesh_decomposer.process(
                self._simulation_information, state_template_dir, self._mesh_filename
            )

    def read_full_field_data(self, filename):
        """
        Configure model to read full field data compatible with FieldSeriesData importer.
        """
        self._set_results_reader_object(FieldSeriesData)
        self.set_results_filename(filename)


class _MatcalGeneratedSierraModelBase(SierraModelBase):
    """
    Base class for MatCal-generated SIERRA models that produce their own results.
    """

    def set_results_filename(self, filename, file_type=None):
        raise AttributeError(
            "Calling 'sets_results_filename' is not allowed for MatCal standard models."
        )


class _StandardSierraModelBase(_MatcalGeneratedSierraModelBase):
    """
    Base model for MatCal generated SIERRA models, not intended for users.
    """

    _input_file_class = SierraFileBase

    class TemperatureFieldNotPresentError(RuntimeError):
        pass

    @property
    @abstractmethod
    def _death_blocks(self):
        """"""

    @abstractmethod
    def _geometry_creator_class(self):
        """"""

    @abstractmethod
    def _get_loading_boundary_condition_displacement_function(self, state, params_by_precedent):
        """"""

    @abstractmethod
    def _create_user_output_blocks(self, state):
        """"""

    @abstractmethod
    def _additional_boundary_condition_setup(self, state):
        """"""

    @property
    @abstractmethod
    def _loading_bc_node_sets(self):
        """"""

    @property
    @abstractmethod
    def _loading_bc_directions(self):
        """"""

    @property
    @abstractmethod
    def _fixed_bc_node_sets(self):
        """"""

    @property
    @abstractmethod
    def _fixed_bc_directions(self):
        """"""

    @property
    @abstractmethod
    def _model_blocks(self):
        """"""

    @property
    def input_file(self):
        """
        SIERRA input file object associated with this model.
        """
        return self._input_file

    def _revise_derived_names(self):
        self._input_filename = matcal_name_format(self._name) + ".i"
        self._mesh_filename = matcal_name_format(self._name) + ".g"
        self._input_file._set_local_mesh_filename(self._mesh_filename)

    def _check_material(self, material):
        if not isinstance(material, Material):
            raise TypeError(
                "Materials passed to a standard model must be of type Material. "
                f"Passed {material} which is of type {type(material)}."
            )

    def __init__(self, material, executable="adagio", **geo_params):
        self._check_material(material)
        self._material = material
        self._input_file = self._input_file_class(self._material, self._death_blocks)
        super().__init__(executable=executable)

        self._assign_material_parameters()

        self._base_geo_params = self._geometry_creator_class.Parameters(**geo_params)
        self._current_state_geo_params = None

        self._input_file._set_fixed_boundary_conditions(
            self._fixed_bc_node_sets, self._fixed_bc_directions
        )

        self._boundary_condition_data = DataCollection("boundary conditions")
        self._boundary_condition_scale_factor = 1.0
        self._temperature_field_from_boundary_data = None

    def set_boundary_condition_scale_factor(self, value):
        """
        Scales the dependent and independent field in the model deformation function
        by a constant factor. It must be between 1 and 10.
        """
        check_value_is_real_between_values(value, 1, 10, "value", closed=True)
        self._boundary_condition_scale_factor = value

    def add_boundary_condition_data(self, data):
        """
        Add boundary condition data (Data or DataCollection) used to determine
        loading functions for each state.
        """
        if isinstance(data, DataCollection):
            self._boundary_condition_data += data
        elif isinstance(data, Data):
            self._boundary_condition_data.add(data)
        else:
            raise TypeError(
                "Expected a data collection or data importer for add_boundary_condition_data. "
                f"Received object of type {type(data)}."
            )

    def read_temperature_from_boundary_condition_data(self, field_name=TEMPERATURE_KEY):
        """
        Read and apply a temperature history from boundary condition data.
        """
        check_value_is_nonempty_str(field_name, "field_name")
        self._temperature_field_from_boundary_data = field_name

    def reset_boundary_condition_data(self):
        """
        Remove all previously added boundary condition data and clear any request to
        read temperature from that data.

        This resets the model's boundary-condition data so new boundary
        condition data can be added before the next run/setup.
        """
        self._boundary_condition_data = DataCollection("boundary conditions")
        self._temperature_field_from_boundary_data = None

    def _get_parameters_by_precedence(self, state):
        model_constants = self.get_model_constants(state)
        params_by_precedence = _get_parameters_according_to_precedence(state, model_constants)
        param_source_by_precedence = _get_parameters_source_according_to_precedence(
            state, model_constants
        )
        return params_by_precedence, param_source_by_precedence

    def _setup_state(self, state, state_template_dir=".", build_mesh=True):
        params_by_precedence, param_source = self._get_parameters_by_precedence(state)

        self._update_geometry_parameters(params_by_precedence, param_source)
        self._input_file._reset_state_boundary_conditions_and_output()

        self._set_state_loading_boundary_condition(state)
        self._additional_boundary_condition_setup(state)
        self._set_state_model_temperature(state)
        self._create_user_output_blocks(state)

        if build_mesh:
            self._prepare_mesh(state_template_dir, state)

        self._prepare_template_files(state_template_dir)

        aprepro = AddApreproParamFileLinesPreprocessor()
        aprepro.process(template_dir=state_template_dir, input_filename=self._input_filename)

    def _prepare_mesh(self, state_template_dir, state):
        mesh_filename = os.path.join(state_template_dir, self._mesh_filename)
        self._generate_mesh(state_template_dir, mesh_filename, state)
        self._decompose_mesh(state_template_dir, mesh_filename)

    def _generate_mesh(self, state_template_dir, mesh_filename, state):
        mesh_generator = self._geometry_creator_class(
            mesh_filename=mesh_filename, geometry_parameters=self._current_state_geo_params
        )
        mesh_generator.create_mesh(template_dir=state_template_dir)

    def _decompose_mesh(self, state_template_dir, mesh_filename):
        mesh_preparer = DecomposeAndCopyMeshPreprocessor()
        mesh_preparer.process(
            computing_info=self._simulation_information,
            template_dir=state_template_dir,
            mesh_filename=mesh_filename,
            delete_source_mesh=True,
        )

    def _update_geometry_parameters(self, params_by_precedent, param_source):
        self._current_state_geo_params = OrderedDict(self._base_geo_params.parameters)

        def format_val_str(val):
            if isinstance(val, str):
                return f"\"{val}\""
            return f"{val}"

        for param, value in params_by_precedent.items():
            for geometry_param in self._current_state_geo_params.keys():
                if geometry_param == param:
                    logger.info(
                        f'\t\tUpdating geometry parameter "{geometry_param}" '
                        f"to {param_source[param]} value {format_val_str(value)}"
                    )
                    self._current_state_geo_params[param] = value

        param_class = self._geometry_creator_class.Parameters
        self._current_state_geo_params = param_class(**self._current_state_geo_params)

    def _prepare_template_files(self, template_dir):
        logger.info(f'\t\tWriting SIERRA input deck "{self._input_filename}".')
        input_filename = os.path.abspath(os.path.join(template_dir, self._input_filename))
        self._input_file.write_input_to_file(input_filename)
        logger.info("\t\tInput deck complete.\n")

    def _check_boundary_conditions_added(self):
        if self._boundary_condition_data.state_names == []:
            raise RuntimeError(
                f'No model boundary condition data for model "{self._name}" has been added.'
            )

    def _check_state_in_boundary_condition_data(self, state):
        if state.name not in self._boundary_condition_data.state_names:
            raise KeyError(
                f'The state "{state.name}" is not in the model boundary condition data.'
            )

    def _prepare_loading_boundary_condition_displacement_function(self, state, params_by_precedent):
        self._check_boundary_conditions_added()
        self._check_state_in_boundary_condition_data(state)
        return self._get_loading_boundary_condition_displacement_function(
            state, params_by_precedent
        )

    def _set_state_loading_boundary_condition(self, state):
        params_by_precedent, _src = self._get_parameters_by_precedence(state)
        bc_func = self._prepare_loading_boundary_condition_displacement_function(
            state, params_by_precedent
        )

        # BC comment hook: derived classes may set _last_loading_bc_comment; default None
        bc_comment = getattr(self, "_last_loading_bc_comment", None)

        self._input_file._add_prescribed_loading_boundary_condition_with_displacement_function(
            bc_func,
            self._loading_bc_node_sets,
            self._loading_bc_directions,
            self._loading_bc_direction_keys,
            self._boundary_condition_scale_factor,
            bc_comment=bc_comment,
        )

    def _set_state_model_temperature(self, state):
        self._input_file._reset_state_temperature_conditions()
        if self._temperature_field_from_boundary_data is not None:
            boundary_data_fields = self._boundary_condition_data.state_field_names(state.name)
            temperature_in_data = self._temperature_field_from_boundary_data in boundary_data_fields
            time_in_data = TIME_KEY in boundary_data_fields

            if temperature_in_data and time_in_data:
                bc_data = self._boundary_condition_data
                temp_key = self._temperature_field_from_boundary_data
                self._input_file._set_state_prescribed_temperature_from_boundary_data(
                    bc_data, state, temp_key
                )
            elif not temperature_in_data:
                raise self.TemperatureFieldNotPresentError(
                    f"The field '{self._temperature_field_from_boundary_data}' is "
                    "not in the boundary condition "
                    f"DataCollection for state '{state}'. Check input for model '{self.name}'."
                )
            elif not time_in_data:
                raise self.TemperatureFieldNotPresentError(
                    f"The field '{TIME_KEY}' is not in the boundary condition "
                    f"DataCollection for state '{state}' "
                    "and is required for a temperature based on boundary condition data. "
                    f"Check input for model '{self.name}'."
                )
        else:
            params_by_precedent, _parameter_source = self._get_parameters_by_precedence(state)
            self._input_file._set_initial_temperature_from_parameters(params_by_precedent)

    def _set_last_loading_bc_comment(self, metadata, extra_lines=None):
        if extra_lines is None:
            extra_lines = []
        self._last_loading_bc_comment = "\n".join(
            format_bc_function_comment_lines(metadata) + extra_lines
        )

    @property
    def coupling(self):
        """
        Returns the type of thermomechanical coupling that the model will use in a simulation. Returns None if uncoupled.
        """
        return self._input_file.coupling

    @property
    def exodus_output(self):
        """
        Returns True if exodus output is activated and variables have been added. Otherwise, returns False.
        """
        return self._input_file.exodus_output_active

    def set_number_of_time_steps(self, number_of_steps):
        """
        Set the number of load/time steps used in the SIERRA procedure. 
        Due to other model options and adaptive time stepping, 
        this number of time steps is not guaranteed.

        :param number_of_steps: Total number of time steps in the analysis.
        :type number_of_steps: int
        """
        check_value_is_positive_integer(number_of_steps, "number_of_steps")
        self._input_file._set_number_of_time_steps(number_of_steps)

    def set_end_time(self, end_time):
        """   
        Set the analysis termination time. This will override the final time
        determined from the supplied boundary condition and state data.
        
        This is most useful when boundary condition data is a complex load history 
        and only a portion of it needs to be simulated. 
        It may also be useful when trying to simulate stress relaxation after the end of loading.

        :param end_time: Final simulation time.
        :type end_time: float
        """
        check_item_is_correct_type(end_time, numbers.Real, "end_time")
        self._input_file._set_end_time(end_time)

    def set_start_time(self, start_time):
        """
        Set the analysis start time. This will override the start time
        determined from the supplied boundary condition and state data.

        This is most useful when boundary condition data is a 
        complex load history and only a portion of it needs to be simulated.

        :param start_time: Initial simulation time.
        :type start_time: float
        """
        check_item_is_correct_type(start_time, numbers.Real, "start_time")
        self._input_file._set_start_time(start_time)

    def use_total_lagrange_element(self):
        """
        Sets the model to use SIERRA/SM’s total lagrange 
        8 node hexahedral element with volume average J turned on.
        """
        self._input_file._use_total_lagrange_element()

    def use_under_integrated_element(self):
        """
        Sets the model to use SIERRA/SM’s under integrated element 
        8 node hexahedral element with hourglass control and the
        SIERRA/SM default settings for the element.
        """
        self._input_file._use_under_integrated_element()
        self._base_geo_params.update({"element_type": "hex8"})

    def activate_thermal_coupling(self):
        """
        Activates adiabatic heating for the model. 
        An initial temperature is added to the model and 
        additional temperature outputs are provided in the heartbeat and exodus output. 
        """
        self._verify_temperature_not_read_from_boundary_data()
        self._input_file._activate_adiabatic_heating()

    def _verify_temperature_not_read_from_boundary_data(self):
        if self._temperature_field_from_boundary_data is not None:
            raise RuntimeError(
                f"Model '{self.name}' cannot activate coupling and prescribe "
                "a temperature from boundary data."
            )

    def add_nodal_output_variable(self, *nodal_variable_names):
        """
        Add nodal output variables for the model. 
        The method accepts a comma separated list of strings. 
        These should be valid nodal variables for the model and material model being used.

        :param nodal_variable_names: One or more nodal variable names to output.
        :type nodal_variable_names: list(str)
        """
        self._input_file._add_nodal_output_variable(*nodal_variable_names)

    def add_element_output_variable(self, *element_variable_names, volume_average=True):
        """
        Add element output variables for the model. 
        The method accepts a comma separated list of strings. 
        These should be valid element variables for the model and material model being used.

        :param element_variable_names: One or more element variable names to output.
        :type element_variable_names: list(str)

        :param volume_average: If True, output a volume-averaged form of each element
            variable when applicable.
        :type volume_average: bool  
        """
        self._input_file._add_element_output_variable(
            *element_variable_names, volume_average=volume_average
        )

    def activate_exodus_output(self, output_step_interval=20):
        """
        Enable exodus mesh/results output.

        :param output_step_interval: Write exodus output every
            ``output_step_interval`` steps.
        :type output_step_interval: int
        """
        check_value_is_positive_integer(output_step_interval, "output_step_interval")
        self._input_file._activate_exodus_output(output_step_interval)

    @property
    def element_type(self):
        """
        The element type being used by the model.
        """
        return self._input_file.element_type

    def _assign_material_parameters(self):
        self._input_file._add_solid_mechanics_finite_element_parameters(
            self._material.name, self._material.model, *self._model_blocks
        )

    def set_minimum_timestep(self, minimum_timestep):
        """
        Sets a minimum timestep such that the simulation will 
        exit cleanly if the timestep is cut to below the user specified minimum value. 
        This is done using SIERRA/SM solution termination.

        :param minimum_timestep: Minimum acceptable timestep.
        :type minimum_timestep: float
        """
        check_value_is_positive_real(minimum_timestep, "minimum_timestep")
        sol_term = self.input_file.solution_termination
        sol_term.add_global_termination_criteria("timestep", minimum_timestep, "<")

    def set_convergence_tolerance(
        self,
        target_relative_residual,
        target_residual=None,
        acceptable_relative_residual=None,
        acceptable_residual=None,
    ):
        """
        Set the convergence tolerance values for the SIERRA/SM conjugate gradient solver. 
        By default the target residual is two orders of magnitude higher than the target 
        relative residual, and the acceptable relative residual is one order of magnitude 
        higher than the target relative residual. Updating the target relative residual 
        will update the target residual and acceptable relative residual according 
        to these defaults. No acceptable residual is specified by default.

        :param target_relative_residual: the relative residual for convergence of the 
            SIERRA/SM conjugate gradient solver. Must be between zero and one.
        :type target_relative_residual: float

        :param target_residual: the target residual for convergence of the 
            SIERRA/SM conjugate gradient solver. Must be positive.
        :type target_residual: float or None

        :param acceptable_relative_residual: the acceptable relative residual for 
            convergence of the SIERRA/SM conjugate gradient solver. 
            Must be positive and greater than the target relative residual but less than one.
        :type acceptable_relative_residual: float or None

        :param acceptable_residual: the acceptable residual for convergence of 
            the SIERRA/SM conjugate gradient solver. 
            Must be positive and  should be greater than the target residual.
        :type acceptable_residual: float or None
        """
        self._input_file._set_cg_convergence_tolerance(
            target_relative_residual,
            target_residual,
            acceptable_relative_residual,
            acceptable_residual,
        )
        check_value_is_real_between_values(
            target_relative_residual, 0, 
            1, "target_relative_residual"
        )
        if target_residual is not None:
            check_value_is_positive_real(
                target_residual, "target_residual"
            )
        if acceptable_relative_residual is not None:
            check_value_is_real_between_values(
                acceptable_relative_residual, target_relative_residual, 
                1, "acceptable_relative_residual"
            )
        if acceptable_residual is not None:
            check_value_is_positive_real(
                acceptable_residual, "acceptable_residual"
            )


class _StandardSierraModelWithDeathBase(_StandardSierraModelBase):
    def activate_element_death(self, death_variable="damage", critical_value=0.15):
        """
        Activate element death based on a damage/failure variable.

        :param death_variable: Element variable used to determine failure.
        :type death_variable: str

        :param critical_value: Threshold value at which an element is deleted/deactivated.
        :type critical_value: float or str
        """
        check_value_is_nonempty_str(death_variable, "death_variable")
        check_item_is_correct_type(critical_value, (numbers.Real, str), "critical_value")
        self._input_file._activate_element_death(death_variable, critical_value)

    @property
    def failure(self):
        """
        Returns the type of failure that the model will use in a simulation.
        Returns None if there is no failure.
        """ 
        return self._input_file.failure


class _CoupledStandardSierraModelBase(_StandardSierraModelWithDeathBase):
    _input_file_class = SierraFileWithCoupling

    def __init__(self, material, executable="adagio", **kwargs):
        super().__init__(material, executable, **kwargs)
        self._input_file._set_thermal_bc_nodesets(self._thermal_bc_nodesets)

    @property
    @abstractmethod
    def _thermal_bc_nodesets(self):
        """"""

    @property
    @abstractmethod
    def _temperature_blocks(self):
        """"""

    def use_composite_tet_element(self):
        """
        Use the composite tetrahedral total-Lagrange element formulation.

        This is a convenience wrapper around :meth:`use_total_lagrange_element`
        with ``use_composite_tet=True``.
        """
        self.use_total_lagrange_element(use_composite_tet=True)

    def use_total_lagrange_element(self, use_composite_tet=False):
        """
        Sets the model to use SIERRA/SM’s total lagrange 
        8 node hexahedral element with volume average J turned on. Or, optionally, 
        use composite tetrahedral elements for the model.

        :param use_composite_tet: If True, use the composite tetrahedral version of
            the formulation; otherwise use the hex formulation.
        :type use_composite_tet: bool        
        """
        self._input_file._use_total_lagrange_element(use_composite_tet)
        if use_composite_tet:
            self._base_geo_params.update({"element_type": "tet10"})
        else:
            self._base_geo_params.update({"element_type": "hex8"})

    def activate_thermal_coupling(
        self,
        thermal_conductivity=None,
        density=None,
        specific_heat=None,
        plastic_work_variable=None,
        executable="arpeggio",
    ):
        """
        Activates thermomechanical coupling for the MatCal generated model.
        If no options are passed, the model assumes adiabatic heating is 
        being added through the material model. For the adiabatic case, 
        an initial temperature is added to the model and 
        additional temperature outputs are provided in the heartbeat and exodus output. 
        
        If the additional input arguments for are provided, then staggered coupling through Arpeggio is used. 
        Staggered coupling is setup to advance the solid mechanics solve, 
        pass the displacements and plastic work to the thermal solver, 
        advance the thermal solve, and pass the temperature to the 
        solid mechanics solver before continuing to the next time step. 
        
        To activate iterative coupling, use the :meth:`use_iterative_coupling` method.

        :param thermal_conductivity: Thermal conductivity for the thermal model.
        :type thermal_conductivity: float or None

        :param density: Density for the thermal model.
        :type density: float or None

        :param specific_heat: Specific heat for the thermal model.
        :type specific_heat: float or None

        :param plastic_work_variable: Element variable name representing plastic work rate 
            passed to the thermal solver as the element volumetric heat source
        :type plastic_work_variable: str or None

        :param executable: Executable used for the coupled solve.
            This can be an optional path to an custom compiled executable
        :type executable: str
        """
        self._verify_temperature_not_read_from_boundary_data()
        if (
            thermal_conductivity is not None
            and density is not None
            and specific_heat is not None
            and plastic_work_variable is not None
        ):
            check_value_is_nonnegative_real(thermal_conductivity, "thermal_conductivity")
            check_value_is_positive_real(density, "density")
            check_value_is_positive_real(specific_heat, "specific_heat")
            check_value_is_nonempty_str(plastic_work_variable, "plastic_work_variable")
            self.set_executable(executable)
            self._input_file._activate_thermal_coupling(
                thermal_conductivity, density, specific_heat, plastic_work_variable
            )
        elif (
            thermal_conductivity is not None
            or density is not None
            or specific_heat is not None
            or plastic_work_variable is not None
        ):
            raise ValueError(
                f'Error activating coupling for model "{self.name}". '
                "Thermal conductivity, density, specific heat and "
                "the plastic work rate variable name "
                "all must be specified to activate loose thermal coupling."
            )
        else:
            self._input_file._activate_adiabatic_heating()

    def use_iterative_coupling(self):
        """
        Activates iterative coupling for the model. Iterative coupling can only be used 
        after thermal coupling with staggered coupling has been activated for the model.
        """
        if self.coupling == _Coupling.staggered:
            self._input_file._activate_iterative_coupling()
        else:
            raise RuntimeError(
                f'Iterative coupling for model "{self.name}" can only be set after staggered '
                'thermomechanical coupling has been activated with ".activate_thermal_coupling"'
            )

    def _add_temperature_global_outputs(self):
        if self.coupling is not None:
            temp_block_str = " ".join(self._temperature_blocks)
            global_temp_output = SolidMechanicsUserOutput(
                "global_temperature_output", temp_block_str, "block"
            )
            self._input_file._solid_mechanics_region.add_subblock(global_temp_output)

            if self.coupling == _Coupling.adiabatic:
                add_global_temp_method = global_temp_output.add_compute_global_from_element_field
            else:
                add_global_temp_method = global_temp_output.add_compute_global_from_nodal_field

            add_global_temp_method("low_temperature", TEMPERATURE_KEY, "min")
            add_global_temp_method("med_temperature", TEMPERATURE_KEY, "average")
            add_global_temp_method("high_temperature", TEMPERATURE_KEY, "max")
            self._input_file._add_heartbeat_global_variable("low_temperature")
            self._input_file._add_heartbeat_global_variable("med_temperature")
            self._input_file._add_heartbeat_global_variable("high_temperature")


class _ThreeDimensionalStandardSierraModelBase(_CoupledStandardSierraModelBase):
    _input_file_class = SierraFileThreeDimensional

    def __init__(self, material, executable="adagio", **geo_params):
        _updated_geo_params = {"element_type": "hex8"}
        _updated_geo_params.update(geo_params)
        super().__init__(material, executable, **_updated_geo_params)
        self._allowable_load_drop_factor = None
        self.set_allowable_load_drop_factor(0.5)
        self._full_field_output = False
        self._nonlocal_radius = None
        self._death_variable = None

    @property
    @abstractmethod
    def _solution_termination_variable(self):
        """"""

    @property
    @abstractmethod
    def _create_derived_user_output_blocks(self):
        """"""

    def use_total_lagrange_element(self, use_composite_tet=False):
        super().use_total_lagrange_element(use_composite_tet)
        self._update_nonlocal_variables()

    def use_under_integrated_element(self):
        super().use_under_integrated_element()
        self._update_nonlocal_variables()

    def _update_nonlocal_variables(self):
        if self._nonlocal_radius is not None and self._death_variable is not None:
            self._input_file._add_nonlocal_user_output(self._death_variable, self._nonlocal_radius)

    def set_allowable_load_drop_factor(self, value):
        """
        The allowable drop in the models "load" field before the simulation is terminated.
        Note that the actual load field name is model dependent. The simulation will terminate
        when the following is true:

        .. math:: load < max\\_load(1-value)

        where max_load is the maximum load in the current load history. The load drop 
        factor must be between 0 and 1.

        :param value: the max allowable load drop fraction
        :type value: float
        """
        check_value_is_real_between_values(value, 0, 1, "allowable_load_drop_factor", closed=True)
        self._allowable_load_drop_factor = value

    def _create_user_output_blocks(self, state):
        self._create_derived_user_output_blocks(state)
        self._add_temperature_global_outputs()
        self._update_nonlocal_variables()
        self._add_solution_termination_user_output(state)
        self._add_full_field_output()

    def _add_full_field_output(self):
        if self._full_field_output:
            results_file_name = self._results_information.results_filename
            self._input_file._activate_full_field_results_output(
                results_file_name, 
                *self._model_blocks
            )

    def _add_solution_termination_user_output(self, state):
        params_by_precedent, source = self._get_parameters_by_precedence(state)
        drop_factor = self._allowable_load_drop_factor
        if "allowable_load_drop_factor" in params_by_precedent:
            drop_factor = params_by_precedent["allowable_load_drop_factor"]
            drop_factor_source = source["allowable_load_drop_factor"]
            logger.info(
                '\t\tUpdating model parameter "allowable_load_drop_factor" to '
                f"{drop_factor_source} value {drop_factor}"
            )
        self._input_file._add_solution_termination_user_output(
            self._solution_termination_variable, drop_factor
        )

    def activate_full_field_data_output(self, full_field_window_width, full_field_window_height):
        """
        Activate full field data output for calibrations requiring full field data. The 
        parameters for this method specify the rectangular window where data will be output. 
        Currently, this is only implemented for the 
        :class:`~matcal.sierra.models.RoundUniaxialTensionModel`, 
        :class:`~matcal.sierra.models.RectangularUniaxialTensionModel` and 
        the :class:`~matcal.sierra.models.RoundNotchedTensionModel`. 
        The rectangular window starts 
        at the axial and radial center of the model and goes outward according to the two 
        arguments passed to this function. Since these models feature 1/8th symmetry, the window 
        only covers one octant of the model. The full field window width and height 
        specify the width and height 
        of the 2D window in the octant where the model is built. 
        Any nodes within that width and height
        of the window will be included in the output. 
        
        .. warning:
            No checks are made on these values. 
            If mesh faces are not found within the specified window,
            the model may error out or not produce mesh 
            output for full field comparisons.

        :param full_field_window_width: width of the 2D window for full field data output. 
            The width is aligned with the X direction of the global coordinate system of the model.
        :type full_field_window_width: float

        :param full_field_window_height: height of the 2D window for full field data output. 
            The height is aligned with the Y direction of the global coordinate system of the model 
            and its axis of loading.
        :type full_field_window_height: float
        """
        check_value_is_positive_real(full_field_window_height, "full_field_window_height")
        check_value_is_positive_real(full_field_window_width, "full_field_window_width")
        self._base_geo_params["full_field_window_width"] = full_field_window_width
        self._base_geo_params["full_field_window_height"] = full_field_window_height
        full_field_results_filename = "results/full_field_results.e"
        self._set_results_reader_object(FieldSeriesData)
        self._results_information.results_filename = full_field_results_filename
        self._full_field_output = True

    def activate_element_death(
            self, death_variable="damage", 
            critical_value=0.15, nonlocal_radius=None
        ):

        """
        Activates element death for the model. It will kill elements that have 
        the element variable with
        name "death_variable" reach the critical value. To use nonlocal damage also 
        specify the "nonlocal_radius" and the "initial_value" for the "death_variable". Both
        options are needed for nonlocal damage to work correctly.

        :param death_variable: the name of the element variable that governs element death.
        :type death_variable: str

        :param critical_value: the element "death_variable" value at which 
            the element will die. Elements 
            with a "death_variable" value less than the "critical_value" are kept alive. This can 
            also be a string if it is replaced by a MatCal design or state parameter on run time. 
            For example, it could be set to "{critical_value}" and the critical value could be 
            a MatCal study parameter.
        :type critical_value: float, str

        :param nonlocal_radius: the radius to be used for nonlocal average 
            in the the geometry units. Specifying 
            this parameter to anything but None activates nonlocal averaging 
            for the "death_variable". This must be greater than zero.
        :type nonlocal_radius: float
        """        
        super().activate_element_death(death_variable, critical_value)
        if nonlocal_radius is not None:
            check_value_is_positive_real(nonlocal_radius, "nonlocal_radius")
            self._nonlocal_radius = nonlocal_radius
            self._death_variable = death_variable
            super().activate_element_death("damage", critical_value)
            self._input_file._add_nonlocal_user_output(death_variable, nonlocal_radius)

    def activate_implicit_dynamics(self):
        """
        Turns on implicit dynamics for SIERRA/SM. By default, all models
        are run quasi-statically.
        """        
        self._input_file._activate_implicit_dynamics()


class _SymmetricUniaxiallyLoadedModelBase(_ThreeDimensionalStandardSierraModelBase):
    _loading_bc_directions = ["y"]
    _loading_bc_direction_keys = ["component"]

    _fixed_bc_node_sets = ["ns_x_symmetry", "ns_y_symmetry", "ns_z_symmetry"]
    _fixed_bc_directions = ["x", "y", "z"]

    _solution_termination_variable = LOAD_KEY

    def _additional_boundary_condition_setup(self, state):
        """"""

    def _add_disp_outputs(self, disp_ns, disp_factor):
        disp_output = SolidMechanicsUserOutput("global_disp", disp_ns, "node set")
        self._input_file._solid_mechanics_region.add_subblock(disp_output)
        disp_output.add_compute_global_from_nodal_field("partial_displacement", "displacement(y)")
        disp_output.add_compute_global_from_expression(
            DISPLACEMENT_KEY, f"partial_displacement*{disp_factor};"
        )
        self._input_file._add_heartbeat_global_variable(DISPLACEMENT_KEY)

    def _add_load_outputs(self, load_ns, load_factor):
        load_output = SolidMechanicsUserOutput("global_load", load_ns, "node set")
        self._input_file._solid_mechanics_region.add_subblock(load_output)
        load_output.add_compute_global_from_nodal_field("partial_load", "reaction(y)", "sum")
        load_output.add_compute_global_from_expression(LOAD_KEY, f"partial_load*{load_factor};")
        self._input_file._add_heartbeat_global_variable(LOAD_KEY)

    def _get_loading_boundary_condition_displacement_function(self, state, params_by_precedent):
        disp_function, metadata = get_displacement_function_from_load_displacement_data_collection(
            self._boundary_condition_data,
            state,
            params_by_precedent,
            scale_factor=1.0,
            return_metadata=True,
        )
        self._set_last_loading_bc_comment(metadata)
        return disp_function


class _SymmetricUniaxiallyLoadedModelContactBase(_SymmetricUniaxiallyLoadedModelBase):
    _input_file_class = SierraFileThreeDimensionalContact

    def activate_self_contact(self, friction_coefficient=0.3):
        """
        Activates self-contact for the model. 

        :param friction_coefficient: the desired friction coefficient for self-contact
        :type friction_coefficient: float
        """
        check_value_is_nonnegative_real(friction_coefficient, "friction_coefficient")
        logger.warning(
            f'Use of self contact with the MatCal generated SIERRA/SM model "{self.name}" '
            "may be unreliable and/or result in long run times."
        )
        self._input_file._activate_self_contact(friction_coefficient)

    def set_contact_convergence_tolerance(
        self,
        target_relative_residual,
        target_residual=None,
        acceptable_relative_residual=None,
        acceptable_residual=None,
    ):

        """
        Set the convergence tolerance values for the control contact block. 
        By default the target residual and acceptable relative residual are 
        set to one order of magnitude higher than the target relative residual. 
        Updating the target 
        relative residual will update the target residual and acceptable relative residual 
        according to these defaults. 

        The conjugate gradient solver settings will also 
        be updated such that its target relative residual is one order of magnitude 
        less than the contact target relative residual, its target residual is 
        one order of magnitude higher than the contact target relative and 
        its acceptable relative residual is set to 10. To specify custom CG solver 
        convergence tolerances with contact, call 
        :meth:`~matcal.sierra.models.UniaxialLoadingMaterialPointModel.set_convergence_tolerance`
        after calling this method.

        No acceptable residual is specified for either solver.

        :param target_relative_residual: the relative residual for convergence 
            of SIERRA/SM control contact. Must be between zero and one.
        :type target_relative_residual: float

        :param target_residual: the target residual for convergence 
            of SIERRA/SM control contact. Must be positive.
        :type target_residual: float

        :param acceptable_relative_residual: the acceptable relative residual for convergence 
            of SIERRA/SM control contact. Must be positive and greater than the target relative 
            residual.
        :type acceptable_relative_residual: float

        :param acceptable_residual: the acceptable residual for convergence 
            of SIERRA/SM control contact. Must be positive and should be greater than the 
            target residual.
        :type acceptable_residual: float
        """
        self._input_file._set_contact_convergence_tolerance(
            target_relative_residual, target_residual, 
            acceptable_relative_residual, acceptable_residual
        )
        check_value_is_real_between_values(
            target_relative_residual, 0, 
            1, "target_relative_residual"
        )
        if target_residual is not None:
            check_value_is_positive_real(
                target_residual, "target_residual"
            )
        if acceptable_relative_residual is not None:
            check_value_is_real_between_values(
                acceptable_relative_residual, target_relative_residual, 
                100, "acceptable_relative_residual"
            )
        if acceptable_residual is not None:
            check_value_is_positive_real(
                acceptable_residual, 
                "acceptable_relative_residual"
            )


