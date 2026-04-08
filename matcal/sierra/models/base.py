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
    check_value_is_nonnegative_real,
)

from matcal.full_field.data_importer import FieldSeriesData

from matcal.sierra.material import Material
from matcal.sierra.input_file_writer import (
    SierraFileBase,
    SierraFileWithCoupling,
    SierraFileThreeDimensional,
    SierraFileThreeDimensionalContact,
    SolidMechanicsUserOutput,
    SolidMechanicsUserVariable,
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

    @property
    def coupling(self):
        return self._input_file.coupling

    @property
    def exodus_output(self):
        return self._input_file.exodus_output_active

    def set_number_of_time_steps(self, number_of_steps):
        check_value_is_positive_integer(number_of_steps, "number_of_steps")
        self._input_file._set_number_of_time_steps(number_of_steps)

    def set_end_time(self, end_time):
        check_item_is_correct_type(end_time, numbers.Real, "end_time")
        self._input_file._set_end_time(end_time)

    def set_start_time(self, start_time):
        check_item_is_correct_type(start_time, numbers.Real, "start_time")
        self._input_file._set_start_time(start_time)

    def use_total_lagrange_element(self):
        self._input_file._use_total_lagrange_element()

    def use_under_integrated_element(self):
        self._input_file._use_under_integrated_element()
        self._base_geo_params.update({"element_type": "hex8"})

    def activate_thermal_coupling(self):
        self._verify_temperature_not_read_from_boundary_data()
        self._input_file._activate_adiabatic_heating()

    def _verify_temperature_not_read_from_boundary_data(self):
        if self._temperature_field_from_boundary_data is not None:
            raise RuntimeError(
                f"Model '{self.name}' cannot activate coupling and prescribe "
                "a temperature from boundary data."
            )

    def add_nodal_output_variable(self, *nodal_variable_names):
        self._input_file._add_nodal_output_variable(*nodal_variable_names)

    def add_element_output_variable(self, *element_variable_names, volume_average=True):
        self._input_file._add_element_output_variable(
            *element_variable_names, volume_average=volume_average
        )

    def activate_exodus_output(self, output_step_interval=20):
        check_value_is_positive_integer(output_step_interval, "output_step_interval")
        self._input_file._activate_exodus_output(output_step_interval)

    @property
    def element_type(self):
        return self._input_file.element_type

    def _assign_material_parameters(self):
        self._input_file._add_solid_mechanics_finite_element_parameters(
            self._material.name, self._material.model, *self._model_blocks
        )

    def set_minimum_timestep(self, minimum_timestep):
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
            check_value_is_real_between_values(
                target_residual, target_relative_residual, 
                1, "target_residual"
            )
        if acceptable_relative_residual is not None:
            check_value_is_real_between_values(
                acceptable_relative_residual, target_relative_residual, 
                1, "acceptable_relative_residual"
            )
        if acceptable_residual is not None:
            check_value_is_real_between_values(
                acceptable_residual, self._input_file._cg.get_target_residual(), 
                1, "acceptable_residual"
            )


class _StandardSierraModelWithDeathBase(_StandardSierraModelBase):
    def activate_element_death(self, death_variable="damage", critical_value=0.15):
        check_value_is_nonempty_str(death_variable, "death_variable")
        check_item_is_correct_type(critical_value, (numbers.Real, str), "critical_value")
        self._input_file._activate_element_death(death_variable, critical_value)

    @property
    def failure(self):
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
        self.use_total_lagrange_element(use_composite_tet=True)

    def use_total_lagrange_element(self, use_composite_tet=False):
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
        super().activate_element_death(death_variable, critical_value)
        if nonlocal_radius is not None:
            check_value_is_positive_real(nonlocal_radius, "nonlocal_radius")
            self._nonlocal_radius = nonlocal_radius
            self._death_variable = death_variable
            super().activate_element_death("damage", critical_value)
            self._input_file._add_nonlocal_user_output(death_variable, nonlocal_radius)

    def activate_implicit_dynamics(self):
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
        disp_function = get_displacement_function_from_load_displacement_data_collection(
            self._boundary_condition_data, state, params_by_precedent, scale_factor=1.0
        )
        # Optional: set comment for input deck if you implement comment builder upstream
        self._last_loading_bc_comment = None
        return disp_function


class _SymmetricUniaxiallyLoadedModelContactBase(_SymmetricUniaxiallyLoadedModelBase):
    _input_file_class = SierraFileThreeDimensionalContact

    def activate_self_contact(self, friction_coefficient=0.3):
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
        self._input_file._set_contact_convergence_tolerance(
            target_relative_residual, target_residual, 
            acceptable_relative_residual, acceptable_residual
        )
        check_value_is_real_between_values(
            target_relative_residual, 0, 
            1, "target_relative_residual"
        )
        if target_residual is not None:
            check_value_is_real_between_values(
                target_residual, target_relative_residual, 
                1, "target_residual"
            )
        if acceptable_relative_residual is not None:
            check_value_is_real_between_values(
                acceptable_relative_residual, target_relative_residual, 
                100, "acceptable_relative_residual"
            )
        if acceptable_residual is not None:
            check_value_is_real_between_values(
                acceptable_residual, 
                self._input_file._contact_target_residual, 
                100, 
                "acceptable_relative_residual"
            )


