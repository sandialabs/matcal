"""
This module contains all the MatCal tools to create and 
use SIERRA models with MatCal studies.
"""

from abc import abstractmethod
from collections import OrderedDict
from glob import glob
import numbers
import numpy as np
import os
import shutil

from matcal.core.boundary_condition_calculators import (
    get_displacement_function_from_load_displacement_data_collection, 
    get_displacement_function_from_strain_data_collection, 
    raise_required_fields_not_found_error, 
    get_rotation_function_from_data_collection, BoundaryConditionDeterminationError)
from matcal.core.constants import (DESIGN_PARAMETER_FILE,  DISPLACEMENT_KEY, DISPLACEMENT_RATE_KEY, 
    ENG_STRAIN_KEY, ENG_STRESS_KEY, LOAD_KEY, ROTATION_KEY, STATE_PARAMETER_FILE, STRAIN_RATE_KEY, 
    TEMPERATURE_KEY, TIME_KEY,TORQUE_KEY, TRUE_STRAIN_KEY, TRUE_STRESS_KEY)
from matcal.core.data import Data, DataCollection
from matcal.core.logger import initialize_matcal_logger
from matcal.core.models import (  
    AdditionalFileCopyPreprocessor, InputFileCopyPreprocessor, ModelBase, 
    ModelPreprocessorBase, _copy_file_or_directory_to_target_directory, 
    _get_mesh_template_folder)

from matcal.core.parameters import (get_parameters_according_to_precedence, 
    get_parameters_source_according_to_precedence)

from matcal.core.utilities import (matcal_name_format, check_value_is_nonempty_str, 
    check_value_is_real_between_values, check_item_is_correct_type, check_value_is_positive_integer, 
    check_value_is_positive_real, _convert_list_of_files_to_abs_path_list,)

from matcal.cubit.geometry import (MaterialPointGeometry, RectangularUniaxialTensionGeometry, 
    RoundUniaxialTensionGeometry, RoundNotchedTensionGeometry, SolidBarTorsionGeometry, 
    TopHatShearGeometry)

from matcal.exodus.geometry import ExodusHexGeometryCreator
from matcal.exodus.mesh_modifications import ExodusHex8MeshExploder, _ExodusFieldInterpPreprocessor

from matcal.full_field.data import FieldData
from matcal.full_field.data_importer import FieldSeriesData, ImportedTwoDimensionalMesh
from matcal.full_field.field_mappers import MeshlessMapperGMLS
from matcal.full_field.TwoDimensionalFieldGrid import FieldGridBase

from matcal.core.mesh_modifications import get_mesh_decomposer


from matcal.sierra.material import Material
from matcal.sierra.input_file_writer import (AnalyticSierraFunction, SierraFileBase, SierraFileWithCoupling, 
    SierraFileThreeDimensional, SolidMechanicsUserOutput, SolidMechanicsUserVariable, 
    SierraFileThreeDimensionalContact, _Coupling)                                   
from matcal.sierra.simulators import SierraSimulator


logger = initialize_matcal_logger(__name__)


class _AddApreproParamFileLinesPreprocessor(ModelPreprocessorBase):        
    def __init__(self):
        self.param_aprepro_include = f'{{include({DESIGN_PARAMETER_FILE})}}\n'
        self.state_aprepro_include = f'{{include({STATE_PARAMETER_FILE})}}\n'    

    def process(self, template_dir, input_filename):
        input_filename = os.path.basename(input_filename)
        input_file =  f"{template_dir}/{input_filename}"
        _add_aprepro_to_input(input_file, self.param_aprepro_include)
        _add_aprepro_to_input(input_file, self.state_aprepro_include)


class _DecomposeAndCopyMeshPreprocessor(ModelPreprocessorBase):
    """
    Model preprocessor not intended for users.
    """

    def __init__(self):
        super().__init__()
        
    def process(self, computing_info, template_dir, mesh_filename, 
                delete_source_mesh=False):
        mesh_decomposer_class = get_mesh_decomposer(mesh_filename)
        mesh_decomposer = mesh_decomposer_class()
        logger.info(f"\t\tPreparing mesh \"{os.path.split(mesh_filename)[-1]}\"")
        n_cores = computing_info.number_of_cores
        mesh_files_template_folder = _get_mesh_template_folder(template_dir)
        template_mesh_filename = os.path.join(mesh_files_template_folder,
                                            os.path.basename(mesh_filename))
        
        logger.debug(f"\t\tThe path to the mesh is:\n{template_mesh_filename}\n")
        if delete_source_mesh:
            shutil.move(mesh_filename, template_mesh_filename)
            mesh_filename=template_mesh_filename
        if n_cores > 1:
            mesh_decomposer.decompose_mesh(os.path.abspath(mesh_filename), 
                                                 n_cores, mesh_files_template_folder)
            if os.path.exists(template_mesh_filename):
                os.remove(template_mesh_filename)
        else:
            _copy_file_or_directory_to_target_directory(mesh_files_template_folder, mesh_filename)
        for file in glob(mesh_files_template_folder+os.path.sep+"*"):
            src_file = os.path.abspath(file)
            dest_file = os.path.join(template_dir, os.path.basename(file))
            if src_file != dest_file:
                os.symlink(src_file, dest_file)
        logger.info(f"\t\tMesh ready")


def _add_aprepro_to_input(filename, message):
    temp_file = filename+".temp"
    with open(filename, 'r') as f_read:
        with open(temp_file, 'w') as f_write:
            f_write.write(message)
            for line in f_read:
                f_write.write(line)
    shutil.copy(temp_file, filename)
    os.remove(temp_file)


class _SierraModelBaseNew(ModelBase):
    _simulator_class = SierraSimulator

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._additional_executable_arguments = []
        self._check_syntax = False
        self._check_input = False
        
    def _get_simulator_class_inputs(self, state):
        args = [self.name, self._simulation_information, self._results_information, 
                state, self._input_filename]
        kwargs = {"custom_commands":self._additional_executable_arguments,
                 "check_syntax":self._check_syntax, 
                "check_input":self._check_input, 
                "epu_results":self._epu_results(), 
                "model_constants":self.get_model_constants(state)}

        return args, kwargs

    def _epu_results(self):
        exodus_reader = (self._results_information.results_reader_object ==
                                   FieldSeriesData)
        parallel = self._simulation_information.number_of_cores > 1
        if exodus_reader and parallel:
            return True
        else:
            return False

    def add_executable_argument(self, argument):
        """Use this method to pass additional arguments 
        to the SIERRA executable (adagio, aria, .etc). 
        No checks are performed on their validity. This should 
        be called multiple times to add multiple arguments. 
        For example, to add "--aprepro on" to the arguments list for 
        the executable, call this once for "--aprepro" and again for "on".
        
        :param argument: the argument to be passed to the executable. 
            This will be passed directly, as provided, to the executable. 

        :type argument: str
        """
        if not isinstance(argument, str):
            message = (f"Sierra Flags need to be passed as strings.\n" +
                      f" Flag Passed: {argument}\nFlag Type: {type(argument)}")
            raise TypeError(message)
        self._additional_executable_arguments.append(argument)

    def run_check_input(self, state, parameter_collection, target_directory=None):
        """ Runs the SIERRA executable with the "--check-input" option. Use 
        for debugging your SIERRA models before running a more in-depth study. 
        The arguments and returns are the same as the 
        :meth:`~matcal.sierra.models.UserDefinedSierraModel.run` method
        except no results data are returned. 

        This method will check the input syntax, verify the mesh input and
        run most model initializations for the SIERRA executable. 
        """
        self._check_input = True
        results = super().run(state, parameter_collection, target_directory)
        self._check_input = False
        return results

    def run_check_syntax(self, state, parameter_collection, target_directory=None):
        """ Runs the SIERRA executable with the "--check-syntax" option. Use 
        for debugging your SIERRA models before running a more in-depth study. 
        The arguments and returns are the same as the 
        :meth:`~matcal.sierra.models.UserDefinedSierraModel.run` method 
        except no results data are returned.

        This method will only check the input syntax but runs quickly. 
        """
        self._check_syntax = True
        results = super().run(state, parameter_collection, target_directory)
        self._check_syntax = False
        return results


class UserDefinedSierraModel(_SierraModelBaseNew):  
    """
    The UserDefinedSierraModel class allows users to use their 
    own models in MatCal calibrations. To create a user
    specified model, the user must supply as stand-alone input 
    deck for their model and the mesh file for their model.
    """
    model_type = "user_defined_sierra_model"

    def __init__(self, executable, simulation_input_file, 
                 simulation_mesh_filename, *other_sources):
        """
        :param executable: Name of the sierra executable the user wishes to run.
        :type executable: str

        :param simulation_input_file: The path to the input file.
        :type simulation_input_file: str

        :param simulation_mesh_filename: The path to the mesh file.
        :type simulation_mesh_filename: str

        :param other_sources: additional files or directories that need to be
            in the working directory of the SIERRA model so that it can
            run. These are include files that the main input file may need.
        :type other_sources: list(str)
        """
        super().__init__(executable=executable)
        self._input_filename = os.path.abspath(simulation_input_file)
        self._mesh_filename = os.path.abspath(simulation_mesh_filename)
        other_sources = _convert_list_of_files_to_abs_path_list(other_sources)
        self._additional_sources_to_copy = other_sources
        
    def _setup_state(self, state, state_template_dir='.', 
            build_mesh=True):
        ifile_copier = InputFileCopyPreprocessor()
        ifile_copier.process(state_template_dir, input_filename=self._input_filename)
        
        aprepro_preprocessor = _AddApreproParamFileLinesPreprocessor()
        aprepro_preprocessor.process(state_template_dir, input_filename=self._input_filename)
        
        additional_file_copier = AdditionalFileCopyPreprocessor()
        additional_file_copier.process(state_template_dir, self._additional_sources_to_copy)
        if build_mesh:
            mesh_decomposer = _DecomposeAndCopyMeshPreprocessor()
            mesh_decomposer.process(self._simulation_information, state_template_dir, 
                                    self._mesh_filename)
   
    def read_full_field_data(self, filename):
        """
        Allows the model to read in full field data. The model
        expects a filename compatible with the
        :func:`~matcal.full_field.data_importer.FieldSeriesData` 
        data importer.

        :param filename: The full field results filename for the model.
        :type filename: str
        """
        self._set_results_reader_object(FieldSeriesData)
        self.set_results_filename(filename)


class _MatcalGeneratedSierraModelNew(_SierraModelBaseNew):
   
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def set_results_filename(self, filename, file_type=None):
        """
        .. warning::
            Not a valid function for MatCal generated models that produce their own 
            results file. Calling this will result in an error.
        """

        raise AttributeError("Calling \'sets_results_filename\' is not "
                                      "allowed for MatCal standard models.")
 

class _StandardSierraModelNew(_MatcalGeneratedSierraModelNew):

    """
    Base model for MatCal generated SIERRA models, not intended for users.
    """

    _input_file_class = SierraFileBase

    class TemperatureFieldNotPresentError(RuntimeError):
        def __init__(self, *args):
             super().__init__(*args)

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
        """Returns the input file object associated with the 
        model. See :mod:`matcal.sierra.input_file_writer` module."""
        return self._input_file

    def _revise_derived_names(self):
        self._input_filename = matcal_name_format(self._name) + ".i"

        self._mesh_filename = matcal_name_format(self._name) + ".g"
        self._input_file._set_local_mesh_filename(self._mesh_filename)

    def _check_material(self, material):
        if not isinstance(material, Material):
            raise TypeError(
            "Materials passed to a standard model must be of "
            f"type Material. Passed {material} which is of type {type(material)}.")

    def __init__(self, material, executable="adagio", **geo_params):
        self._check_material(material)
        self._material = material
        self._input_file = self._input_file_class(self._material, self._death_blocks)
        super().__init__(executable=executable)
        self._assign_material_parameters()
        self._base_geo_params = self._geometry_creator_class.Parameters(**geo_params)
        self._current_state_geo_params = None
        self._input_file._set_fixed_boundary_conditions(self._fixed_bc_node_sets, 
            self._fixed_bc_directions)
        self._boundary_condition_data = DataCollection('boundary conditions')
        self._boundary_condition_scale_factor = 1.0
        self._temperature_field_from_boundary_data = None

    def set_boundary_condition_scale_factor(self, value):
        """
        Scales the dependent 
        and independent field in the model deformation function 
        by a constant factor. It must be between 1 and 10.

        :param value: scale factor for the model boundary condition function 
        :type value: float
        """
        check_value_is_real_between_values(value, 1, 10, 
           "value", "SierraModel.set_boundary_condition_scale_factor", 
           closed=True)
        self._boundary_condition_scale_factor = value

    def add_boundary_condition_data(self, data):
        """
        Standard models in :mod:`~matcal.sierra.models` require that data
        be passed to the model such the appropriate boundary conditions can be 
        extracted and applied to the MatCal
        generated model for each state of interest for the study. 
        See the documentation on the specific model for more
        information on what data is needed for boundary conditions for that model.

        :param data: Information on the materials being studied for this particular model.

        :type data: :class:`~matcal.core.data.DataCollection` or
                :class:`~matcal.core.data.Data`

        :raises TypeError: If the wrong type is passed for the data.
        """
        if isinstance(data, DataCollection):
            self._add_bound_condition_data_collection(data)
        elif isinstance(data, Data):
            self._add_boundary_condition_data(data)
        else:
            raise TypeError("Expected a data collection or data importer for the "+
                            "add_boundary_condition_data method. "
                            "Received object of type {}.".format(type(data)))
       
    def _add_boundary_condition_data(self, data):
        self._boundary_condition_data.add(data)

    def _add_bound_condition_data_collection(self, data):
        self._boundary_condition_data += data

    def read_temperature_from_boundary_condition_data(self, field_name=TEMPERATURE_KEY):
        f"""
        Allows the model to read and apply a temperature history from the data. The temperature 
        field name and \"{TIME_KEY}\" must exist in the data. 

        :param field_name: optional user specified field name for the temperature.
            Default is "{TEMPERATURE_KEY}".
        :type field_name: str
        """
        check_value_is_nonempty_str(field_name, "field_name", 
                                    "SierraModel.read_temperature_from_boundary_condition_data")
        self._temperature_field_from_boundary_data = field_name

    def reset_boundary_condition_data(self):
        """
        Clears out all boundary condition data previously added to the model. To run the model,
        new boundary condition data will need to be added.
        """
        self._boundary_condition_data = DataCollection('boundary conditions')
        self._temperature_field_from_boundary_data = None

    def _get_parameters_by_precedence(self, state):
        model_constants = self.get_model_constants(state)
        params_by_precedence = get_parameters_according_to_precedence(state, model_constants)
        param_source_by_precedence = get_parameters_source_according_to_precedence(state, 
            model_constants)
        return params_by_precedence, param_source_by_precedence 

    def _setup_state(self, state, state_template_dir='.', build_mesh=True):
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
        aprepro_prepoc = _AddApreproParamFileLinesPreprocessor()
        aprepro_prepoc.process(template_dir=state_template_dir, 
                               input_filename=self._input_filename)
    
    def _prepare_mesh(self, state_template_dir, state):
        mesh_filename = os.path.join(state_template_dir, self._mesh_filename)
        self._generate_mesh(state_template_dir, mesh_filename, state)
        self._decompose_mesh(state_template_dir, mesh_filename)

    def _generate_mesh(self, state_template_dir, mesh_filename, state):
        mesh_generator = self._geometry_creator_class(mesh_filename=mesh_filename,
                        geometry_parameters=self._current_state_geo_params)
        mesh_generator.create_mesh(template_dir=state_template_dir)
        
    def _decompose_mesh(self, state_template_dir, mesh_filename):
        mesh_preparer = _DecomposeAndCopyMeshPreprocessor()
        mesh_preparer.process(computing_info=self._simulation_information, 
            template_dir=state_template_dir, mesh_filename=mesh_filename, delete_source_mesh=True)

    def _update_geometry_parameters(self, params_by_precedent, param_source):
        self._current_state_geo_params = OrderedDict(self._base_geo_params.parameters)
        def format_val_str(val):
            val_str = f"{val}"
            if isinstance(val, str):
               val_str = f"\"{val}\"" 
            return val_str
        for param, value in params_by_precedent.items():
            for geometry_param in self._current_state_geo_params.keys():
                if geometry_param == param:
                    logger.info(f"\t\tUpdating geometry parameter \"{geometry_param}\" "
                                f"to {param_source[param]} value {format_val_str(value)}")
                    self._current_state_geo_params[param] = value
        param_class = self._geometry_creator_class.Parameters
        self._current_state_geo_params = param_class(**self._current_state_geo_params)

    def _prepare_template_files(self, template_dir):
        logger.info(f"\t\tWriting SIERRA input deck \"{self._input_filename}\".")
        input_filename = os.path.abspath(os.path.join(template_dir, self._input_filename))
        self._input_file.write_input_to_file(input_filename)
        logger.info(f"\t\tInput deck complete.\n")

    def _check_boundary_conditions_added(self):
        if self._boundary_condition_data.state_names == []:      
            raise RuntimeError(f"No model boundary condition data for model "
                f"\"{self._name}\" has been added. Check input and add "
                " the appropriate states to the model boundary condition data.")

    def _check_state_in_boundary_condition_data(self, state):
        if state.name not in self._boundary_condition_data.state_names:      
            error_str = (f"The state \"{state.name}\" is not in the model boundary "+
                         "condition data for model \"{self._name}\". Check input "+
                         "and add the state to the model boundary condition data.")
            raise KeyError(error_str)

    def _prepare_loading_boundary_condition_displacement_function(self, state, 
                                                                  params_by_precedent):
        self._check_boundary_conditions_added()
        self._check_state_in_boundary_condition_data(state)
        disp_function = self._get_loading_boundary_condition_displacement_function(state, 
            params_by_precedent)
        return disp_function
    
    def _set_state_loading_boundary_condition(self, state):
        params_by_precedent, source_by_precedence = self._get_parameters_by_precedence(state)
        bc_func = self._prepare_loading_boundary_condition_displacement_function(state, 
            params_by_precedent)
        ifile = self._input_file
        ifile._add_prescribed_loading_boundary_condition_with_displacement_function(bc_func, 
            self._loading_bc_node_sets, self._loading_bc_directions, 
            self._loading_bc_direction_keys, self._boundary_condition_scale_factor)

    def _set_state_model_temperature(self, state):
        self._input_file._reset_state_temperature_conditions()
        if self._temperature_field_from_boundary_data is not None:
            boundary_data_fields = self._boundary_condition_data.state_field_names(state.name)
            temperature_in_data = (self._temperature_field_from_boundary_data in 
                                   boundary_data_fields)
            time_in_data = TIME_KEY in boundary_data_fields
            if temperature_in_data and time_in_data:
                bc_data = self._boundary_condition_data
                temp_key = self._temperature_field_from_boundary_data
                self._input_file._set_state_prescribed_temperature_from_boundary_data(bc_data,
                    state, temp_key)
            elif not temperature_in_data:
                err_str = (f"The field \'{self._temperature_field_from_boundary_data}\' "+
                           "is not in the boundary condition "+ 
                           f"DataCollection for state \'{state}\'. "+
                           f"Check input for model \'{self.name}\'")
                raise self.TemperatureFieldNotPresentError(err_str)
            elif not time_in_data:
                err_str = (f"The field \'{TIME_KEY}\' is not in the boundary condition "+ 
                           f"DataCollection for state \'{state}\' and is required for "+
                           "a temperature based on the boundary condition data. "
                           f"Check input for model \'{self.name}\'")
                raise self.TemperatureFieldNotPresentError(err_str)
        else:
            params_by_precedent, parameter_source = self._get_parameters_by_precedence(state)
            self._input_file._set_initial_temperature_from_parameters(params_by_precedent)

    @property
    def coupling(self):
        """Returns the type of thermomechanical coupling that the model 
        will use in a simulation. Returns None if uncoupled."""
        return self._input_file.coupling

    @property
    def exodus_output(self):
        """Returns True if exodus output is activated and variables have been added.
        Otherwise, returns False."""
        return self._input_file.exodus_output_active

    def set_number_of_time_steps(self, number_of_steps):
        """
        Sets the target number of time steps for the simulation to take. 
        Due to other model options and adaptive time stepping, this number 
        of time steps is not guaranteed.

        :param number_of_steps: the desired number of simulation time steps
        :type number_of_steps: int
        """

        check_value_is_positive_integer(number_of_steps, "number_of_steps", 
                                         "SierraModel.set_number_of_time_steps")
        self._input_file._set_number_of_time_steps(number_of_steps)

    def set_end_time(self, end_time):
        """
        Sets the simulation end time. This is most useful when boundary condition 
        data is a complex load history and only a portion of it needs to be simulated. 
        It may also be useful when trying to simulate stress relaxation after the end of 
        loading.

        :param start_time: the simulation end time
        :type start_time: float
        """
        check_item_is_correct_type(end_time, numbers.Real, "end_time", 
                                         "SierraModel.set_end_time")
        self._input_file._set_end_time(end_time)

    def set_start_time(self, start_time):
        """
        Sets the simulation start time. This is most useful when boundary condition 
        data is a complex load history and only a portion of it needs to be simulated.

        :param start_time: the simulation start time
        :type start_time: float
        """
        check_item_is_correct_type(start_time, numbers.Real, "start_time", 
                                         "SierraModel.start_end_time")
        self._input_file._set_start_time(start_time)

    def use_total_lagrange_element(self):
        """
        Sets the model to use SIERRA/SM's total lagrange 
        8 node hexahedral element with default settings.
        """
        self._input_file._use_total_lagrange_element()

    def use_under_integrated_element(self):
        """
        Sets the model to use SIERRA/SM's under 
        integrated element 8 node hexahedral element 
        with hourglass control with default settings.
        """
        self._input_file._use_under_integrated_element()
        self._base_geo_params.update({"element_type":"hex8"})

    def activate_thermal_coupling(self):
        """
        Activates adiabatic heating for the material point. 
        By definition, a material point cannot support 
        coupling with conduction. 
        """
        self._verify_temperature_not_read_from_boundary_data()
        self._input_file._activate_adiabatic_heating()

    def _verify_temperature_not_read_from_boundary_data(self):
        if self._temperature_field_from_boundary_data is not None:
            raise RuntimeError(f"Model '{self.name}' cannot activate coupling and "
                               "prescribe a temperature from boundary data.")

    def add_nodal_output_variable(self, *nodal_variable_names):
        """
        Add nodal output variables for the model. The method accepts a comma separated list 
        of strings. These should be valid nodal variables for the model and material model 
        being used.

        :parameter nodal_variable_names: comma separated list of strings 
            for valid nodal variable names
        :type nodal_variable_names: list(str)
        """
        self._input_file._add_nodal_output_variable(*nodal_variable_names)

    def add_element_output_variable(self, *element_variable_names, 
                                    volume_average=True):
        """
        Add element output variables for the model. The method accepts a comma separated list 
        of strings. These should be valid element variables for the model and material model 
        being used.

        :param element_variable_names: comma separated list of strings for 
            valid element variable names
        :type element_variable_names: list(str)

        :param volume_average: volume average the quantity, default is True 
        :type volume_average: bool
        """
        self._input_file._add_element_output_variable(*element_variable_names, 
            volume_average=volume_average)

    def activate_exodus_output(self, output_step_interval=20):
        """
        Turns on exodus results output for the model with an output interval of every 20 steps 
        by default. The output_step_interval argument can be used to modify the output interval.

        :param output_step_interval: the desired output step interval
        :type output_step_interval: int
        """
        check_value_is_positive_integer(output_step_interval, "output_step_interval",
            "SierraModel.activate_exodus_output")
        self._input_file._activate_exodus_output(output_step_interval)

    @property
    def element_type(self):
        """
        The element type being used by the model."""
        return self._input_file.element_type

    def _assign_material_parameters(self):
        self._input_file._add_solid_mechanics_finite_element_parameters(self._material.name, 
            self._material.model, *self._model_blocks)

    def set_minimum_timestep(self, minimum_timestep):
        """
        Sets a minimum timestep such that the simulation will exit cleanly 
        if the timestep is cut to below the user specified minimum value.
        This is done using SIERRA/SM solution termination.

        :param minimum_timestep: the minimum value allowed for the simulation timestep.
            If the timestep falls below this value due to adaptive timestepping the 
            simulation will be exit cleanly.
        :type minimum_timestep: float
        """
        check_value_is_positive_real(minimum_timestep, "minimum_timestep", 
            "SierraModel.set_minimum_timestep")
        sol_term = self.input_file.solution_termination
        sol_term.add_global_termination_criteria("timestep", minimum_timestep, "<")

    def set_convergence_tolerance(self, target_relative_residual, target_residual=None, 
        acceptable_relative_residual=None, acceptable_residual=None):
        """
        Set the convergence tolerance values for the SIERRA/SM 
        conjugate gradient solver. 
        By default the target residual is two orders of magnitude higher than the 
        target relative residual, and the acceptable relative residual is 
        one order of magnitude higher than the target relative residual. Updating the target 
        relative residual will update the target residual and acceptable relative residual 
        according to these defaults.
        No acceptable residual is specified by default. 

        :param target_relative_residual: the relative residual for convergence 
            of the SIERRA/SM conjugate gradient solver. Must be between zero and one.
        :type target_relative_residual: float

        :param target_residual: the target residual for convergence 
            of the SIERRA/SM conjugate gradient solver. Must be between zero and one.
        :type target_residual: float

        :param acceptable_relative_residual: the acceptable relative residual for convergence 
            of the SIERRA/SM conjugate gradient solver. Must be positive and greater than the
            target relative residual but less than one.
        :type acceptable_relative_residual: float

        :param acceptable_residual: the acceptable residual for convergence 
            of the SIERRA/SM conjugate gradient solver. Must be positive and greater than the 
            target residual but less than one.
        :type acceptable_residual: float
        """
        self._input_file._set_cg_convergence_tolerance(target_relative_residual, target_residual, 
            acceptable_relative_residual, acceptable_residual)
        check_value_is_real_between_values(target_relative_residual, 0, 1, 
            "target_relative_residual", "SierraModel.set_convergence_tolerance")
        if target_residual is not None:
            check_value_is_real_between_values(target_residual, target_relative_residual, 1, 
                "target_residual", "SierraModel.set_convergence_tolerance")
        if acceptable_relative_residual is not None:
            check_value_is_real_between_values(acceptable_relative_residual, 
                target_relative_residual, 1, "acceptable_relative_residual", 
                "SierraModel.set_convergence_tolerance")
        if acceptable_residual is not None:
            check_value_is_real_between_values(acceptable_residual, 
                self._input_file._cg.get_target_residual(), 1, "acceptable_residual", 
                "SierraModel.set_convergence_tolerance")   


class _StandardSierraModelWithDeath(_StandardSierraModelNew):

    def activate_element_death(self, death_variable="damage", critical_value=0.15):
        """
        Activates element death for the model. It will kill elements that have 
        the element variable with
        name "death_variable" reach the critical value.

        :param death_variable: the name of the element variable that governs element death.
        :type death_variable: str

        :param critical_value: the element "death_variable" value at 
            which the element will die. Elements 
            with a "death_variable" value less than the 
            "critical_value" are kept alive. This can 
            also be a string if it is replaced by a MatCal 
            design or state parameter on run time. 
            For example, it could be set to "{critical_value}" and 
            the critical value could be 
            a MatCal study parameter.
        :type critical_value: float, str
        """
        check_value_is_nonempty_str(death_variable, "death_variable", 
                                    "SierraModel.activate_element_death")
        check_item_is_correct_type(critical_value, (numbers.Real, str), 
                                   "SierraModel.activate_element_death",
                                    "critical_value")
        self._input_file._activate_element_death(death_variable, critical_value)

    @property
    def failure(self):
        """Returns the type of failure that the model 
        will use in a simulation. Returns None if there is no failure."""
        return self._input_file.failure


class UniaxialLoadingMaterialPointModel(_StandardSierraModelWithDeath):
    """
    MatCal generated material point model for uniaxial loading.
    """
    model_type = "uniaxial_loading_material_point"
    _geometry_creator_class = MaterialPointGeometry
    _death_blocks = ["material_point_block"]
    _model_blocks = ["material_point_block"]

    _loading_bc_node_sets = ["ns_positive_z"]
    _loading_bc_directions = ["z"]
    _loading_bc_direction_keys = ["component"]

    _fixed_bc_node_sets = ["ns_negative_x",
                           "ns_negative_y", 
                           "ns_negative_z"]
    _fixed_bc_directions = ["x", "y", "z"]
    

    def __init__(self, material):
        super().__init__(material=material, executable="adagio")

    def _additional_boundary_condition_setup(self, state):
        """"""

    def _get_loading_boundary_condition_displacement_function(self, state, params_by_precedent):
        func = get_displacement_function_from_strain_data_collection(self._boundary_condition_data, 
            state, params_by_precedent)
        return func

    def _create_user_output_blocks(self, state):
        self._add_load_outputs()
        self._add_true_stress_strain_outputs()
        self._add_contraction_output()

    def _add_load_outputs(self):
        load_output = SolidMechanicsUserOutput("global_stress_strain_load_disp", 
            "ns_positive_z", "node set")
        self._input_file._solid_mechanics_region.add_subblock(load_output)
        load_output.add_compute_global_from_nodal_field(DISPLACEMENT_KEY, "displacement(z)")
        self._input_file._add_heartbeat_global_variable(DISPLACEMENT_KEY)
        self._input_file._add_heartbeat_global_variable(DISPLACEMENT_KEY, ENG_STRAIN_KEY)
        load_output.add_compute_global_from_nodal_field(LOAD_KEY, "force_external(z)", "sum")
        self._input_file._add_heartbeat_global_variable(LOAD_KEY)
        self._input_file._add_heartbeat_global_variable(LOAD_KEY, ENG_STRESS_KEY)

    def _add_true_stress_strain_outputs(self):
        true_stress_strain_output = SolidMechanicsUserOutput("true_stress_strain", 
                                                             "include all blocks")
        self._input_file._solid_mechanics_region.add_subblock(true_stress_strain_output)
        true_stress_strain_output.add_compute_global_from_element_field(TRUE_STRAIN_KEY, 
                                                                        "log_strain(zz)")
        self._input_file._add_heartbeat_global_variable(TRUE_STRAIN_KEY)
        
        true_stress_strain_output.add_compute_global_from_element_field(TRUE_STRESS_KEY, 
                                                                        "cauchy_stress(zz)")
        self._input_file._add_heartbeat_global_variable(TRUE_STRESS_KEY)

        true_stress_strain_output.add_compute_global_from_element_field("log_strain_xx", 
                                                                        "log_strain(xx)")
        self._input_file._add_heartbeat_global_variable("log_strain_xx")

        true_stress_strain_output.add_compute_global_from_element_field("log_strain_yy", 
                                                                        "log_strain(yy)")
        self._input_file._add_heartbeat_global_variable("log_strain_yy")
        if self.coupling is not None:
            true_stress_strain_output.add_compute_global_from_element_field(TEMPERATURE_KEY, 
                                                                            TEMPERATURE_KEY)
            self._input_file._add_heartbeat_global_variable(TEMPERATURE_KEY)
        
    def _add_contraction_output(self):
        contraction_output = SolidMechanicsUserOutput("contraction", 
            "ns_positive_x", "node set")
        self._input_file._solid_mechanics_region.add_subblock(contraction_output)
        contraction_output.add_compute_global_from_nodal_field("contraction", "displacement(x)")
        self._input_file._add_heartbeat_global_variable("contraction")


class _CoupledStandardSierraModel(_StandardSierraModelWithDeath):
    _input_file_class = SierraFileWithCoupling

    def __init__(self, material, executable="adagio",**kwargs):
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
            self._base_geo_params.update({"element_type":"tet10"})
        else:
            self._base_geo_params.update({"element_type":"hex8"})

    def activate_thermal_coupling(self, thermal_conductivity=None, 
                                    density=None, specific_heat=None, 
                                    plastic_work_variable=None, 
                                    executable="arpeggio"):
        """
        Activates thermomechanical coupling for the MatCal 
        generated model. If no options are passed, 
        the model assumes adiabatic heating is being added 
        through the material model. For the adiabatic case,
        an initial temperature is added to the model 
        and additional temperature 
        outputs are provided in the heartbeat and exodus 
        output. If the additional input arguments for 
        are provided, then staggered coupling through 
        Arpeggio is used. Staggered coupling is setup to 
        advance the solid mechanics solve, 
        pass the displacements and plastic work to 
        the thermal solver, advance
        the thermal solve, and pass the temperature to 
        the solid mechanics solver before continuing to 
        the next time step. To activate iterative coupling, use the 
        :meth:`matcal.sierra.models.RoundUniaxialTensionModel.use_iterative_coupling`
        method. 

        :param thermal_conductivity: the material thermal conductivity
        :type thermal_conductivity: float

        :param density: the material density
        :type density: float

        :param specific_heat: the material specific heat
        :type specific_heat: float

        :param plastic_work_variable: the name of the plastic work rate 
            variable from the material model 
            that will be passed to the thermal solver as 
            the element volumetric heat source.
        :type plastic_work_variable: str

        :param executable: optional path to an custom compiled executable. 
        :type executable: str
        """
        self._verify_temperature_not_read_from_boundary_data()
        if (thermal_conductivity is not None and density is not None
        and specific_heat is not None  and plastic_work_variable is not None):
            check_value_is_positive_real(thermal_conductivity, "thermal_conductivity", 
                                             "SierraModel.activate_thermal_coupling")
            check_value_is_positive_real(density, "density", 
                                             "SierraModel.activate_thermal_coupling")
            check_value_is_positive_real(specific_heat, "specific_heat", 
                                             "SierraModel.activate_thermal_coupling")
            check_value_is_nonempty_str(plastic_work_variable, "plastic_work_variable", 
                                             "SierraModel.activate_thermal_coupling")
            self.set_executable(executable)
            self._input_file._activate_thermal_coupling(thermal_conductivity, density, 
                                                    specific_heat, plastic_work_variable)   
        elif (thermal_conductivity is not None or density is not None  
              or specific_heat is not None or plastic_work_variable is not None):
            err_str = (f"Error activating coupling for model \"{self.name}\". " +
                "Thermal conductivity, density, specific heat and the "+
                "plastic work rate variable name all must be specified "+
                "to activate loose thermal coupling.")
            raise ValueError(err_str)
        else:
            self._input_file._activate_adiabatic_heating()

    def use_iterative_coupling(self):
        """
        Activates iterative coupling for the model. Iterative coupling can only 
        be used after staggered coupling has been activated for the model.
        """

        if self.coupling == _Coupling.staggered:
            self._input_file._activate_iterative_coupling()
        else:
            raise RuntimeError(f"Iterative coupling for model \"{self.name}\" "
                               "can only be set after staggered "
                                "thermomechanical coupling has been activated "
                                "with \".activate_thermal_coupling\"")

    def _add_temperature_global_outputs(self):
        if self.coupling is not None:
            temp_block_str = " ".join(self._temperature_blocks)
            global_temp_output = SolidMechanicsUserOutput("global_temperature_output",
                temp_block_str, "block")
            self._input_file._solid_mechanics_region.add_subblock(global_temp_output)
            if self.coupling == _Coupling.adiabatic:
                add_global_temp_method = global_temp_output.add_compute_global_from_element_field
            elif self.coupling == _Coupling.iterative or self.coupling == _Coupling.staggered:
                add_global_temp_method = global_temp_output.add_compute_global_from_nodal_field
            add_global_temp_method("low_temperature", TEMPERATURE_KEY, "min")
            add_global_temp_method("med_temperature", TEMPERATURE_KEY, "average")
            add_global_temp_method("high_temperature", TEMPERATURE_KEY, "max")
            self._input_file._add_heartbeat_global_variable("low_temperature")
            self._input_file._add_heartbeat_global_variable("med_temperature")
            self._input_file._add_heartbeat_global_variable("high_temperature")


class _ThreeDimensionalStandardSierraModel(_CoupledStandardSierraModel):
    _input_file_class = SierraFileThreeDimensional

    def __init__(self, material, executable="adagio", **geo_params):
        _updated_geo_params = {"element_type":"hex8"}
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
            check_value_is_real_between_values(value, 0, 1, "allowable_load_drop_factor",
                "SierraModel.set_alowable_load_drop_factor", closed=True)
            self._allowable_load_drop_factor = value

    def _create_user_output_blocks(self, state):
        self._create_derived_user_output_blocks(state)
        self._add_temperature_global_outputs()
        self._update_nonlocal_variables()
        self._add_solution_termination_user_output(state)
        self._add_full_field_output()

    def _add_full_field_output(self):
        if self._full_field_output:
            results_file_name  = self._results_information.results_filename
            self._input_file._activate_full_field_results_output(results_file_name,
                                                                  *self._model_blocks)
        
    def _add_solution_termination_user_output(self, state):
        params_by_precedent, source = self._get_parameters_by_precedence(state)
        drop_factor = self._allowable_load_drop_factor
        if "allowable_load_drop_factor" in params_by_precedent:
            drop_factor = params_by_precedent["allowable_load_drop_factor"]
            drop_factor_source = source["allowable_load_drop_factor"]
            logger.info(f"\t\tUpdating model parameter \"allowable_load_drop_factor\" to "
                        f"{drop_factor_source} value {drop_factor}")
        self._input_file._add_solution_termination_user_output(self._solution_termination_variable, 
                                                   drop_factor)

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
        check_value_is_positive_real(full_field_window_height, "full_field_window_height", 
                                     "SierraModel.activate_full_field_data_output")
        check_value_is_positive_real(full_field_window_width, "full_field_window_width", 
                                     "SierraModel.activate_full_field_data_output")
        self._base_geo_params["full_field_window_width"] = full_field_window_width
        self._base_geo_params["full_field_window_height"] = full_field_window_height
        full_field_results_filename = "results/full_field_results.e"
        self._set_results_reader_object(FieldSeriesData)
        self._results_information.results_filename = full_field_results_filename
        self._full_field_output = True

    def activate_element_death(self, death_variable="damage", critical_value=0.15, 
                               nonlocal_radius=None):
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
            check_value_is_positive_real(nonlocal_radius, "nonlocal_radius", 
                                         "SierraModel.activate_element_death")
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


class _SymmetricUniaxiallyLoadedModelBase(_ThreeDimensionalStandardSierraModel):
    _loading_bc_directions = ["y"]
    _loading_bc_direction_keys = ["component"]

    _fixed_bc_node_sets = ["ns_x_symmetry",
                           "ns_y_symmetry", 
                           "ns_z_symmetry"]
    _fixed_bc_directions = ["x", "y", "z"]
    _solution_termination_variable = LOAD_KEY

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _additional_boundary_condition_setup(self, state):
        """"""

    def _add_disp_outputs(self, disp_ns, disp_factor):
        disp_output = SolidMechanicsUserOutput("global_disp", 
            disp_ns, "node set")
        self._input_file._solid_mechanics_region.add_subblock(disp_output)
        disp_output.add_compute_global_from_nodal_field("partial_displacement", "displacement(y)")
        disp_output.add_compute_global_from_expression(DISPLACEMENT_KEY, 
            f"partial_displacement*{disp_factor};")
        self._input_file._add_heartbeat_global_variable(DISPLACEMENT_KEY)

    def _add_load_outputs(self, load_ns, load_factor):
        load_output = SolidMechanicsUserOutput("global_load", 
            load_ns, "node set")
        self._input_file._solid_mechanics_region.add_subblock(load_output)
        load_output.add_compute_global_from_nodal_field("partial_load", "reaction(y)", "sum")
        load_output.add_compute_global_from_expression(LOAD_KEY, f"partial_load*{load_factor};")
        self._input_file._add_heartbeat_global_variable(LOAD_KEY)

    def _get_loading_boundary_condition_displacement_function(self, state, params_by_precedent):
        disp_func_calculator = get_displacement_function_from_load_displacement_data_collection
        disp_function = disp_func_calculator(self._boundary_condition_data, state, 
            params_by_precedent, scale_factor=1.0)
        return disp_function


class _TensionDerivedModelBase(_SymmetricUniaxiallyLoadedModelBase):
    _loading_bc_node_sets = ["ns_side_grip"]

    _death_blocks = ["necking_section"]
    _model_blocks = ["grip_section", "gauge_section", "necking_section"]
    _temperature_blocks = ["gauge_section", "necking_section"]
    _thermal_bc_nodesets = ["ns_side_grip"]


    def __init__(self, material, executable="adagio", **kwargs):
        _all_input_params = {"mesh_method":3}
        _all_input_params.update(kwargs)
        super().__init__(material, executable=executable, **_all_input_params)

    def _create_derived_user_output_blocks(self, state):
        self._add_disp_outputs("extensometer_surf", 2)
        self._add_load_outputs(self._loading_bc_node_sets[0], 4)

 
class _UniaxialTensionModelBase(_TensionDerivedModelBase):
 
    def _get_loading_boundary_condition_displacement_function(self, state, params_by_precedent):
        bc_data=self._boundary_condition_data
        if STRAIN_RATE_KEY in params_by_precedent.keys():
            disp_rate = (params_by_precedent[STRAIN_RATE_KEY]*
                         (self._current_state_geo_params.extensometer_length))
            params_by_precedent.update({DISPLACEMENT_RATE_KEY:disp_rate})
        elif DISPLACEMENT_RATE_KEY in params_by_precedent.keys():
            eng_strain_rate = (params_by_precedent[DISPLACEMENT_RATE_KEY]/
                               self._current_state_geo_params.extensometer_length)
            params_by_precedent.update({STRAIN_RATE_KEY:eng_strain_rate})
        common_state_field_names = bc_data.state_common_field_names(state.name)
        if DISPLACEMENT_KEY in common_state_field_names:
            disp_func_calculator = get_displacement_function_from_load_displacement_data_collection
            scale_factor = (self._current_state_geo_params.gauge_length/
                            self._current_state_geo_params.extensometer_length)
        elif ENG_STRAIN_KEY in common_state_field_names:
            disp_func_calculator = get_displacement_function_from_strain_data_collection
            scale_factor = self._current_state_geo_params.gauge_length
        else:
            raise_required_fields_not_found_error(state, DISPLACEMENT_KEY+", "+ENG_STRAIN_KEY,
                                                  bc_data.name)
        disp_function = disp_func_calculator(bc_data, state, 
            params_by_precedent, scale_factor=scale_factor)
        #Accounting for symmetry across the gauge length
        disp_function[DISPLACEMENT_KEY] *= 0.5
        return disp_function

    def _create_derived_user_output_blocks(self, state):
        super()._create_derived_user_output_blocks(state)
        self._add_strain_outputs()
        self._add_stress_outputs()
        self._add_contraction_output()

    def _add_strain_outputs(self):
        strain_output = SolidMechanicsUserOutput("global_strain", 
            "extensometer_surf", "node set")
        self._input_file._solid_mechanics_region.add_subblock(strain_output)
        extensometer_len = self._current_state_geo_params["extensometer_length"]
        strain_output.add_compute_global_from_expression(ENG_STRAIN_KEY, 
            f"{DISPLACEMENT_KEY}/{extensometer_len};")
        self._input_file._add_heartbeat_global_variable(ENG_STRAIN_KEY)

    def _add_stress_outputs(self):
        stress_output = SolidMechanicsUserOutput("global_stress", 
            "ns_side_grip", "node set")
        self._input_file._solid_mechanics_region.add_subblock(stress_output)
        reference_area = self.reference_area
        stress_output.add_compute_global_from_expression(ENG_STRESS_KEY, 
            f"{LOAD_KEY}/{reference_area};")
        self._input_file._add_heartbeat_global_variable(ENG_STRESS_KEY)

    @property
    @abstractmethod
    def reference_area(self):
        """"""
    @abstractmethod
    def _add_contraction_output(self):
        """"""


class RoundUniaxialTensionModel(_UniaxialTensionModelBase):
    """
    MatCal generated SIERRA/SM uniaxial tension test model with a round cross section.
    """
    model_type = "round_uniaxial_tension_model"
    _geometry_creator_class = RoundUniaxialTensionGeometry

    def _add_contraction_output(self):
        contraction_output_z = SolidMechanicsUserOutput("z_contraction", 
            "z_radial_node", "node set")
        self._input_file._solid_mechanics_region.add_subblock(contraction_output_z)
        contraction_output_z.add_compute_global_from_nodal_field("z_radial_node_z_displacement", 
            "displacement(z)")
        contraction_output_z.add_compute_global_from_expression("z_contraction", 
            "2*z_radial_node_z_displacement;")
        self._input_file._add_heartbeat_global_variable("z_contraction")

        contraction_output_x = SolidMechanicsUserOutput("x_contraction", 
            "x_radial_node", "node set")
        self._input_file._solid_mechanics_region.add_subblock(contraction_output_x)
        contraction_output_x.add_compute_global_from_nodal_field("x_radial_node_x_displacement", 
            "displacement(x)")
        contraction_output_x.add_compute_global_from_expression("x_contraction", 
            "2*x_radial_node_x_displacement;")
        self._input_file._add_heartbeat_global_variable("x_contraction")

    @property
    def reference_area(self):
        r = self._current_state_geo_params["gauge_radius"]
        return np.pi * np.double(r) ** 2

      
class RectangularUniaxialTensionModel(_UniaxialTensionModelBase):
    """
    MatCal generated SIERRA/SM uniaxial tension test model with a rectangular cross section.
    """
    model_type = "rectangular_uniaxial_tension_model"
    _geometry_creator_class = RectangularUniaxialTensionGeometry

    def _add_contraction_output(self):
        contraction_output_z = SolidMechanicsUserOutput("z_contraction", 
            "thickness_center_node", "node set")
        self._input_file._solid_mechanics_region.add_subblock(contraction_output_z)
        contraction_output_z.add_compute_global_from_nodal_field("thickness_node_z_displacement", 
            "displacement(z)")
        contraction_output_z.add_compute_global_from_expression("z_contraction", 
            "2*thickness_node_z_displacement;")
        self._input_file._add_heartbeat_global_variable("z_contraction")

        contraction_output_x = SolidMechanicsUserOutput("x_contraction", 
            "gauge_width_center_node", "node set")
        
        self._input_file._solid_mechanics_region.add_subblock(contraction_output_x)
        contraction_output_x.add_compute_global_from_nodal_field("width_node_x_displacement", 
            "displacement(x)")
        contraction_output_x.add_compute_global_from_expression("x_contraction", 
            "2*width_node_x_displacement;")
        self._input_file._add_heartbeat_global_variable("x_contraction")

    @property
    def reference_area(self):
        return (self._current_state_geo_params["gauge_width"]*
                self._current_state_geo_params["thickness"])


class RoundNotchedTensionModel(_TensionDerivedModelBase):
    """
    MatCal generated SIERRA/SM notched tension test model with a round cross section.
    """
    model_type = "round_notched_tension_model"
    _geometry_creator_class = RoundNotchedTensionGeometry

    def _get_loading_boundary_condition_displacement_function(self, state, params_by_precedent):
        disp_function = super()._get_loading_boundary_condition_displacement_function(state, 
                                    params_by_precedent)
        # Account for symmetry accross gauge section
        disp_function[DISPLACEMENT_KEY] *= 0.5
        return disp_function


class SolidBarTorsionModel(_TensionDerivedModelBase):
    """
    MatCal generated SIERRA/SM solid bar torsion test model.
    """
    model_type = "solid_bar_torsion_model"
    _geometry_creator_class = SolidBarTorsionGeometry

    _loading_bc_node_sets = ["ns_side_grip"]
    _loading_bc_directions = ["cylindrical_axis"]
    _loading_bc_direction_keys = ["cylindrical axis"]
    
    _fixed_bc_node_sets = ["ns_y_symmetry"]
    _fixed_bc_directions = ["y"]

    _solution_termination_variable = TORQUE_KEY

    def _additional_boundary_condition_setup(self, state):
        ifile = self._input_file
        ifile._add_prescribed_displacement_boundary_condition("sierra_constant_function_zero", 
            self._fixed_bc_node_sets, self._loading_bc_directions, 
            self._loading_bc_direction_keys)

    def _create_derived_user_output_blocks(self, state):
        self._add_rotation_user_variables()
        torque_rotation_output = SolidMechanicsUserOutput("global_torque_rotation", 
            "ns_side_grip", "node set")
        self._input_file.solid_mechanics_region.add_subblock(torque_rotation_output, replace=True)
        self._add_variable_transforms(torque_rotation_output)
        self._add_torque_calculations(torque_rotation_output)
        self._add_rotation_calculations(torque_rotation_output)

    def _add_rotation_user_variables(self):
        quarter_rotation_count_var = SolidMechanicsUserVariable("quarter_rotation_count", "global", 
            "real", 0)
        self._input_file.solid_mechanics_region.add_subblock(quarter_rotation_count_var,
            replace=True)

        grip_rotation_var = SolidMechanicsUserVariable("grip_rotation", "global", 
            "real", 0)
        self._input_file.solid_mechanics_region.add_subblock(grip_rotation_var, replace=True)
        
        half_grip_rotation_var = SolidMechanicsUserVariable("add_grip_half_rotation", "global", 
            "real", 0)
        self._input_file.solid_mechanics_region.add_subblock(half_grip_rotation_var, replace=True)

        previous_quadrant_var = SolidMechanicsUserVariable("previous_quadrant", "global", 
            "real", 0)
        self._input_file.solid_mechanics_region.add_subblock(previous_quadrant_var, replace=True)

        current_quadrant_var = SolidMechanicsUserVariable("current_quadrant", "global", 
            "real", 1)
        self._input_file.solid_mechanics_region.add_subblock(current_quadrant_var, replace=True)

    def _add_variable_transforms(self, torque_rotation_output):
        torque_rotation_output.add_nodal_variable_transformation("displacement", 
            "cylindrical_displacement", "cylindrical_coordinate_system")
        torque_rotation_output.add_nodal_variable_transformation("force_external", 
            "cylindrical_force_external", "cylindrical_coordinate_system")

    def _add_torque_calculations(self, torque_rotation_output):
        torque_rotation_output.add_compute_global_from_nodal_field(f"partial_{TORQUE_KEY}", 
            "cylindrical_force_external(y)", "sum")
        grip_radius = self._current_state_geo_params["grip_radius"]
        torque_rotation_output.add_compute_global_from_expression(TORQUE_KEY, 
            f"partial_{TORQUE_KEY}*{grip_radius}")  
        self._input_file._add_heartbeat_global_variable(TORQUE_KEY)

    def _add_rotation_calculations(self, torque_rotation_output):
        torque_rotation_output.add_compute_global_from_nodal_field("grip_cylindrical_x_disp", 
            "cylindrical_displacement(x)")
        torque_rotation_output.add_compute_global_from_nodal_field("grip_cylindrical_y_disp", 
            "cylindrical_displacement(y)")

        torque_rotation_output.add_compute_global_as_function("applied_rotation_radians", 
            self._input_file._load_bc_function_name)
        torque_rotation_output.add_compute_global_from_expression("applied_rotation", 
            "applied_rotation_radians*180/{PI}*2")    

        grip_radius = self._current_state_geo_params["grip_radius"]
        torque_rotation_output.add_compute_global_from_expression("tangent_denominator", 
            f"{grip_radius} - grip_cylindrical_x_disp")

        torque_rotation_output.add_compute_global_from_expression("previous_quadrant", 
            "current_quadrant")

        current_quadrant_expression = ("(grip_cylindrical_y_disp > -1e-15 ) ? " +
            "((tangent_denominator > -1e-15) ? 1 : 2) : " +
            "((tangent_denominator > -1e-15 ) ? 4 : 3)")
        torque_rotation_output.add_compute_global_from_expression("current_quadrant", 
            current_quadrant_expression)
        torque_rotation_output.add_compute_global_from_expression("add_grip_half_rotation", 
            "((current_quadrant == 2) || (current_quadrant == 4)) ? 180 : 0")

        quarter_rotation_count_exp = ("((previous_quadrant != current_quadrant )) ? "+
                                      "quarter_rotation_count+1 : quarter_rotation_count")
        torque_rotation_output.add_compute_global_from_expression("quarter_rotation_count", 
            quarter_rotation_count_exp)
        
        partial_grip_rotation_exp = "(atan(grip_cylindrical_y_disp/(tangent_denominator))*180/{PI})"
        torque_rotation_output.add_compute_global_from_expression(f"partial_{ROTATION_KEY}", 
            partial_grip_rotation_exp)

        grip_rotation_exp = (f"partial_{ROTATION_KEY}*2+90*2*(quarter_rotation_count) + " +
            "add_grip_half_rotation")
        torque_rotation_output.add_compute_global_from_expression(ROTATION_KEY, 
            grip_rotation_exp)

        self._input_file._add_heartbeat_global_variable("applied_rotation")
        self._input_file._add_heartbeat_global_variable(ROTATION_KEY)

    def _get_loading_boundary_condition_displacement_function(self, state, params_by_precedent):
        rot_function = get_rotation_function_from_data_collection(self._boundary_condition_data, state, 
            params_by_precedent, scale_factor=1.0)
        #Accounting for symmetry across the gauge length 
        #and conversion of degrees to radians
        rot_function[ROTATION_KEY] *= 0.5*np.pi/180.0
        rot_function.rename_field(ROTATION_KEY, DISPLACEMENT_KEY)
        return rot_function


class _SymmetricUniaxiallyLoadedModelContactBase(_SymmetricUniaxiallyLoadedModelBase):

    _input_file_class = SierraFileThreeDimensionalContact

    def activate_self_contact(self, friction_coefficient=0.3):
        """
        Activates self-contact for the model. 

        :param friction_coefficient: the desired friction coefficient for self-contact
        :type friction_coefficient: float
        """
        check_value_is_positive_real(friction_coefficient, "friction_coefficient", 
            "SierraModel.activate_self_contact")
        logger.warning(f"Use of self contact with the MatCal generated "
                       f"SIERRA/SM model \"{self.name}\" may be unreliable "
                       f"and/or result in long run times.")
        self._input_file._activate_self_contact(friction_coefficient)

    def set_contact_convergence_tolerance(self, target_relative_residual, target_residual=None, 
        acceptable_relative_residual=None, acceptable_residual=None):
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
            of SIERRA/SM control contact. Must be between zero and one.
        :type target_residual: float

        :param acceptable_relative_residual: the acceptable relative residual for convergence 
            of SIERRA/SM control contact. Must be postive and greater than the target relative 
            residual.
        :type acceptable_relative_residual: float

        :param acceptable_residual: the acceptable residual for convergence 
            of SIERRA/SM control contact. Must be postive and greater than the 
            target residual.
        :type acceptable_residual: float
        """
        self._input_file._set_contact_convergence_tolerance(target_relative_residual, target_residual, 
            acceptable_relative_residual, acceptable_residual)
        check_value_is_real_between_values(target_relative_residual, 0, 1, 
            "target_relative_residual", "SierraModel.set_contact_convergence_tolerance")
        if target_residual is not None:
            check_value_is_real_between_values(target_residual, target_relative_residual, 1, 
                "target_residual", "SierraModel.set_contact_convergence_tolerance")
        if acceptable_relative_residual is not None:
            check_value_is_real_between_values(acceptable_relative_residual, 
                target_relative_residual, 100, "acceptable_relative_residual", 
                "SierraModel.set_contact_convergence_tolerance")
        if acceptable_residual is not None:
            check_value_is_real_between_values(acceptable_residual, 
                self._input_file._contact_target_residual, 100, "acceptable_relative_residual", 
                "SierraModel.set_contact_convergence_tolerance")   


class TopHatShearModel(_SymmetricUniaxiallyLoadedModelContactBase):
    """
    MatCal generated SIERRA/SM top hat shear test model.
    """
    model_type = "top_hat_shear_model"
    _geometry_creator_class = TopHatShearGeometry

    _death_blocks = ["localization_section"]

    _loading_bc_node_sets = ["ns_y_bottom"]

    _model_blocks = ["localization_section", "platten_interface_section"]
    _temperature_blocks = ["localization_section"]
    _thermal_bc_nodesets = ["ns_y_bottom", "ns_load"]

    _fixed_bc_node_sets = ["ns_x_symmetry",
                           "ns_load", 
                           "ns_z_symmetry"]
    _fixed_bc_directions = ["x", "y", "z"]

    @property
    def self_contact(self):
        """
        Returns True if self contact is on for the model. 
        Otherwise, returns False.
        """
        return self._input_file._self_contact()

    def _create_derived_user_output_blocks(self, state):
        self._add_disp_outputs(self._loading_bc_node_sets[0], 1)
        self._add_load_outputs(self._loading_bc_node_sets[0], 4)

    def activate_full_field_data_output(self):
        """
        Will raise an error for this model. Not yet implemented.
        """
        raise AttributeError(f"Cannot use full field data output with the MatCal "
                             f"top hat shear model \"{self.name}\".")


def _vfm_field_series_data(*args, **kwargs):
    data = FieldSeriesData(*args, **kwargs)
    fields_to_keep = []
    for field in data.field_names:
        if "volume" in field:
            fields_to_keep.append(field)
        elif "first_pk_stress" in field:
            fields_to_keep.append(field)
        elif "centroid" in field:
            fields_to_keep.append(field)
        elif "time" in field:
            fields_to_keep.append(field)
    return data[fields_to_keep]


class _VFMStandardSierraModel(_CoupledStandardSierraModel):
    model_type = "VFM"
    _death_blocks = ["block_main"]
    _model_blocks = ["block_main"]
    _temperature_blocks = ["block_main"]
    _thermal_bc_nodesets = []

    _geometry_creator_class = ExodusHexGeometryCreator

    _loading_bc_node_sets = ["front_node_set", "front_node_set", "back_node_set", "back_node_set"]
    _loading_bc_directions = ["x", "y", "x", "y"]
    _loading_bc_direction_keys = ["component", "component", "component", "component"]
    _loading_bc_read_variables = []

    _fixed_bc_node_sets = ["back_node_set"]
    _fixed_bc_directions = ["z"]


    def __init__(self, material, mesh, thickness):
        """ 
        :param material: The material model to be used in this model.
        :type material: :class:`~matcal.sierra.material.Material`

        :param mesh: the surface mesh filename that will be used for 
            creating the VFM model mesh.
        :type mesh: str

        :param thickness: the thickness of the part being simulated using the 
            VFM model.
        :type thickness: float or int
        """
        self._reference_mesh_grid = self._import_mesh(mesh)
        check_value_is_positive_real(thickness, "thickness", "VFMSierraModel")
        self._thickness = thickness
        super().__init__(material, executable="adagio", thickness=self._thickness, 
                         reference_mesh_grid=self._reference_mesh_grid)
        self._input_file._clear_default_element_output_field_names()
        self._set_results_reader_object(_vfm_field_series_data)
        self._results_information.results_filename = os.path.join("results","results.e")
        self._x_displacement_field_name = "U"
        self._y_displacement_field_name = "V"
        self._build_read_variables_list()

        self._polynomial_order = None
        self._search_radius_multiplier = None

        self.set_mapping_parameters()
        self.activate_exodus_output()
        self.add_element_output_variable("first_pk_stress")
        self.add_element_output_variable("centroid", volume_average=False)

    def _build_read_variables_list(self):
        self._loading_bc_read_variables = []
        self._loading_bc_read_variables.append(self._x_displacement_field_name)
        self._loading_bc_read_variables.append(self._y_displacement_field_name)
        self._loading_bc_read_variables.append(self._x_displacement_field_name)
        self._loading_bc_read_variables.append(self._y_displacement_field_name)

    def _create_user_output_blocks(self, state):
        self._add_temperature_global_outputs()

    def _set_state_loading_boundary_condition(self, state):
        params_by_precedent, source_by_precedence = self._get_parameters_by_precedence(state)
        bc_func = self._prepare_loading_boundary_condition_displacement_function(state, 
            params_by_precedent)
        ifile = self._input_file
        ifile._set_time_parameters_to_loading_function(bc_func, scale_factor=1)

    def _additional_boundary_condition_setup(self, state):
        ifile = self._input_file
        ifile._add_prescribed_displacement_boundary_condition(None, self._loading_bc_node_sets,
            self._loading_bc_directions, self._loading_bc_direction_keys, self._loading_bc_read_variables)

    def _import_mesh(self, mesh):
        if isinstance(mesh, str) and os.path.exists(mesh):
            return ImportedTwoDimensionalMesh(mesh)
        elif isinstance(mesh, FieldGridBase):
            return mesh
        else:
            raise FileNotFoundError(f"Could not find the mesh file named \"{mesh}\". Check input.")

    def get_thickness(self):
        """
        Returns the thickness of the model as provided by the use on instantiation.
        """
        return self._thickness

    def set_mapping_parameters(self, polynomial_order: int=
                               MeshlessMapperGMLS.default_polynomial_order, 
                               search_radius_multiplier: float=
                               MeshlessMapperGMLS.default_epsilon_multiplier):
        """
        Set the mapping parameters used by the Compadre GMLS algorithm to map the experimental 
        data onto the mesh. Since extrapolation is frequently needed, by default we use a linear 
        polynomial. Also, a large search radius is used to smooth the data. This maybe undesirable
        if the point cloud is relatively sparse when compared to features of interest.

        :param polynomial_order: the polynomial order to be used for the kernel function in the GMLS 
            algorithm. 
        :type polynomial_order: int

        :param search_radius_multiplier: A multiplier to expand the number of points include while 
            fitting a kernel. 
            This parameter scales the radius needed to find the minimum number of 
            neighbors required to fit the given polynomial and must be greater than or equal 
            to 1. Increasing this parameter increases run time 
            and smooths the data.
        :type search_radius_multiplier: float
         """
        check_value_is_positive_integer(polynomial_order, "polynomial_order", 
                                        "VFMSierraModel.set_mapping_parameters")
        check_value_is_positive_real(search_radius_multiplier, "search_radius_multiplier", 
                                     "VFMSierraModel.set_mapping_parameters")
        self._polynomial_order = polynomial_order
        self._search_radius_multiplier = search_radius_multiplier

    def set_displacement_field_names(self, x_displacement, y_displacement):
        """Set the field names for the displacement fields to be applied 
        to the model from the supplied data series. Default field names 
        are "U" for the X-displacements and "V" for the Y-displacements
        which is standard for DIC.

        .. warning::
           You cannot use the names "displacement_x" and "displacement_y" as this is
           not compatible with SierraSM. Please change the data field names to something
           else if they are so named.
        
        :param x_displacement: the field name for the x-displacement variable
        :type x_displacement: str

        :param y_displacement: the field name for the y-displacement variable
        :type y_displacement: str
        """
        check_value_is_nonempty_str(x_displacement, "x_displacement", 
                                    "VFMSierraModel.set_displacement_field_names")
        check_value_is_nonempty_str(y_displacement, "y_displacement", 
                                    "VFMSierraModel.set_displacement_field_names")
        self._x_displacement_field_name = x_displacement
        self._y_displacement_field_name = y_displacement
        self._build_read_variables_list()

    def add_boundary_condition_data(self, data):
        super().add_boundary_condition_data(data)
        self._verify_valid_boundary_condition_data()
        
    def _verify_valid_boundary_condition_data(self):
        for state_name in self._boundary_condition_data.state_names:
            if len(self._boundary_condition_data[state_name]) > 1:
                raise BoundaryConditionDeterminationError("Currently VFM models only allow one "
                    "boundary condition data set per state. Check input.")
            if not isinstance(self._boundary_condition_data[state_name][0], FieldData):
                error_msg = ("Data passed to VFM models for boundary condition" 
                             " generation must be FieldData types. Received variable"
                             f" of type '{type(self._boundary_condition_data[state_name][0])}'"
                             f" for state '{state_name}'.")
                raise BoundaryConditionDeterminationError(error_msg)

    def  _get_loading_boundary_condition_displacement_function(self, state, params_by_precedent):
        bc_data = self._boundary_condition_data[state.name][0]
        self._verify_fields_in_data(bc_data)
        disp_function_data = bc_data[[TIME_KEY, self._x_displacement_field_name, 
                                      self._y_displacement_field_name]]
        return disp_function_data

    def _get_fields_to_map(self):
        fields_to_project = [self._x_displacement_field_name, self._y_displacement_field_name]
        if self._temperature_field_from_boundary_data is not None:
            fields_to_project.append(self._temperature_field_from_boundary_data)
        return fields_to_project

    def _verify_fields_in_data(self, bc_data):
        fields_to_project = self._get_fields_to_map()
        fields_not_in_data = []
        for field in fields_to_project:
            if field not in bc_data.field_names:
                fields_not_in_data.append(field)
        
        if fields_not_in_data:
            error_msg = ("The correct fields are not in the boundary condition data "
                         f"for state \"{bc_data.state.name}\" for model \"{self.name}\". "
                         f"The data has fields:{bc_data.field_names}\n The following "
                         f"fields were expected: {fields_to_project}")
            raise BoundaryConditionDeterminationError(error_msg)

    def activate_exodus_output(self):
        """
        By default, output is activated and output at every step. The 
        output step interval cannot be adjusted.
        """
        super().activate_exodus_output(output_step_interval=1)

    def set_boundary_condition_scale_factor(self, *args, **kwargs):
        """
        .. warning::
            Not a valid function for MatCal generated VFM models.
            Calling this will result in an error.
        """
        raise AttributeError("Calling \'set_boundary_condition_scale_factor\' is not "
                                      "allowed for MatCal VFM models.")

    def activate_element_death(self, *args, **kwargs):
        """
        .. warning::
            Activating element death is not recommended due to loss of 
            material.
        """
        super().activate_element_death(*args, **kwargs)

    def _generate_mesh(self, state_template_dir, mesh_filename, state):
        super()._generate_mesh(state_template_dir, mesh_filename, state )
        interp_preprocessor = _ExodusFieldInterpPreprocessor()
        interp_preprocessor.process(state_template_dir, mesh_filename,
            self._boundary_condition_data[state][0], fields_to_map=self._get_fields_to_map(), 
            polynomial_order=self._polynomial_order,
            search_radius_multiplier=self._search_radius_multiplier)


class VFMUniaxialTensionHexModel(_VFMStandardSierraModel):
    """
    This model is the MatCal generated Virtual Fields Method SIERRA/SM model 
    for uniaxial tension loading using a linear hex mesh with disconnected elements.
    By default, all elements are disconnected and emulate a plane stress
    material point simulator. The in-plane (x-y plane) degrees of freedom are
    prescribed through interpolation/extrapolation directly from the supplied 
    full-field boundary condition data measured on the surface of the specimen of interest. 
    One in-plane surface on each element is fixed
    in the z direction and the other in-plane surface is free to 
    deform through the thickness. As a result, each element has four degrees 
    of freedom to be solved for on each element - the z-displacements on the free 
    in-plane surface. This requires a finite element solve of the model to determine. However, 
    since the degrees of freedom are significantly reduced from 24 to 4 and 
    completely independent, the simulation cost is generally significantly lower 
    than attempting to solve the entire boundary value problem. This method
    is formulated using a plane stress assumption and will work well for problems 
    that conform well to this assumption. 

    .. include:: vfm_model_notes_and_warnings.rst
    """
        
    def __init__(self, material, surface_mesh_filename, thickness, **kwargs):
        super().__init__(material, surface_mesh_filename, thickness, **kwargs)
        self.add_element_output_variable("volume", volume_average=False)

    def _generate_mesh(self, state_template_dir, mesh_filename, state):
        super()._generate_mesh(state_template_dir, mesh_filename, state)
        mesh_exploder = ExodusHex8MeshExploder(mesh_filename)
        mesh_exploder.boom()

    def activate_thermal_coupling(self, *args, **kwargs):
        """
        Activates only adiabatic heating for the
        disconnected elements used as material point simulators.
        By definition, a material point cannot support 
        coupling with conduction. 
        """
        if args or kwargs:
            raise AttributeError("Calling \'activate_thermal_coupling\' with arguments is not "
                                      "allowed for MatCal VFM disconnected hex models."
                                       " Only adiabatic heating is supported."
                                       " If conduction is desired use the "
                                       "VFMUniaxialTensionConnectedHexModel.")
        super().activate_thermal_coupling()   

    def use_iterative_coupling(self, *args, **kwargs):
        """
        .. warning::
            Not a valid function for the MatCal generated VFM hex model
            with disconnected elements. 
            Calling this will result in an error. Use the 
            :class:`~matcal.sierra.models.VFMUniaxialTensionConnectedHexModel`
        """

        raise AttributeError("Calling \'use_iterative_coupling\' is not "
                            "allowed for this MatCal VFM hex model. Use the "
                            "MatCal VFMUniaxialTensionConnectedHexModel.")


class VFMUniaxialTensionConnectedHexModel(_VFMStandardSierraModel):
    """
    This model is the MatCal generated Virtual Fields Method SIERRA/SM model 
    for uniaxial tension loading using a linear hex mesh with connected elements.
    In this model, all elements are contiguously meshed over the volume of interest with 
    a single element through the half-thickness of the model. With the boundary conditions 
    imposed, the model can approximate a plane stress
    simulation for certain boundary value problems. As with the 
    :class:`~matcal.sierra.models.VFMUniaxialTensionHexModel`, the in-plane degrees of freedom 
    are prescribed through interpolation/extrapolation directly from the supplied 
    full-field boundary condition data. One in-plane surface for the model is fixed
    in the z direction as a symmetry plane and the other in-plane surface is free to 
    deform through the thickness. Solving for this deformation requires a finite element solve of the model.
    However, since the degrees of freedom are significantly reduced in the model,
    the simulation cost is generally significantly lower 
    than attempting to solve the entire boundary value problem. This method
    is formulated using a plane stress assumption and will work well for problems 
    that conform well to this assumption. This contiguously meshed VFM model 
    is provided for problems where solving thermomechanical coupling is desired
    when the heat source is the conversion of plastic work to heat. However, 
    the solution cost may be increased for this method over the disconnected element 
    model. Additionally, the chance that the implicit 
    load steps might not converge to specified tolerances is higher due to the increased 
    difficulty of solving a highly constrained boundary value problem with potentially high
    model form error.

    .. include:: vfm_model_notes_and_warnings.rst
     """

    _geometry_creator_class = ExodusHexGeometryCreator

    def __init__(self, material, surface_mesh_filename, thickness, **kwargs):
        super().__init__(material, surface_mesh_filename, thickness, **kwargs)
        self._thickness /= 2
        self._base_geo_params["thickness"] /= 2.0
        self._double_vol_func = None
        self._double_vol_user_output = None
        self._build_double_volume_input_blocks()

    def _build_double_volume_input_blocks(self):
        double_vol_func = AnalyticSierraFunction("double_volume_func")
        double_vol_func.add_evaluation_expression("2*volume")
        double_vol_func.add_expression_variable("volume", "element", "volume")
        self._double_vol_func = double_vol_func

        double_vol_output = SolidMechanicsUserOutput("double_volume", "include all blocks")
        double_vol_output.add_compute_element_as_function("double_volume", "double_volume_func")
        self._double_vol_user_output = double_vol_output
        
    def _add_double_volume_output(self):
        if self._double_vol_func not in self.input_file.subblocks.values():
            self._input_file.add_subblock(self._double_vol_func)
        sm_region = self.input_file.solid_mechanics_region
        if self._double_vol_user_output not in sm_region.subblocks.values():
            self._input_file._solid_mechanics_region.add_subblock(self._double_vol_user_output)
        if not self._input_file.exodus_output.has_element_output("double_volume", "volume"):
            self._input_file.exodus_output.add_element_output("double_volume", "volume")

    def _create_user_output_blocks(self, state):
        super()._create_user_output_blocks(state)
        self._add_double_volume_output()
    
    def get_thickness(self):
        return self._thickness*2
