"""
The models module includes base classes and helper classes for MatCal models.
It also includes the user facing PythonModel.
"""

from abc import ABC, abstractmethod
from itertools import count
from numbers import Number
import os
from pathlib import Path
import shutil

from matcal.core.computing_platforms import (LocalComputingPlatform, local_computer, 
                                             MatCalPermissionsCheckerFunctionIdentifier,
                                             MatCalComputingPlatformFunctionIdentifier, 
                                             RemoteComputingPlatform)
from matcal.core.constants import (STATE_PARAMETER_FILE, MATCAL_TEMPLATE_DIRECTORY, 
                                   MATCAL_MESH_TEMPLATE_DIRECTORY)
from matcal.core.data_importer import FileData
from matcal.core.parameters import (ParameterCollection, 
                                    get_parameters_according_to_precedence)
from matcal.core.python_function_importer import python_function_importer
from matcal.core.reporter import MatCalParameterReporterIdentifier
from matcal.core.serializer_wrapper import matcal_load
from matcal.core.simulators import PythonSimulator, ExecutableSimulator
from matcal.core.state import  State
from matcal.core.surrogates import _MatCalSurrogateWrapper, MatCalSurrogateBase
from matcal.core.utilities import (make_clean_dir, matcal_name_format, 
                                    check_value_is_nonempty_str, check_item_is_correct_type,
                                    _convert_list_of_files_to_abs_path_list, 
                                    )


from matcal.core.logger import initialize_matcal_logger
logger = initialize_matcal_logger(__name__)


class MeshDecomposer(ABC):
    """
    Base class for mesh decomposers not intended for users.
    """
    def __init__(self):
        self._commands = None
        self._modules_to_load = None
        self._computer = None

    @abstractmethod
    def _build_commands(self, mesh_file, number_of_cores):
        """"""

    @abstractmethod
    def decompose_mesh(self, mesh_file, number_of_cores, output_directory='.',
                        computer=local_computer):
        """"""

    @property
    def modules_to_load(self):
        return self._modules_to_load


class MeshComposer(ABC):
    """
    Base class for mesh composers not intended for users.
    """
    def __init__(self):
        self._commands = None
        self._modules_to_load = None
        self._computer = None

    @abstractmethod
    def _build_commands(self, mesh_file, number_of_cores):
        """"""

    @abstractmethod
    def compose_mesh(self, mesh_file, number_of_cores, output_directory='.', 
                     computer=local_computer):
        """"""

    @property
    def modules_to_load(self):
        return self._modules_to_load


class _DefaultComputeInformation:
    def __init__(self, executable):
        self.computer = local_computer
        self.number_of_cores = 1
        self.time_limit_seconds = None
        self.queue_id = None
        self.modules_to_load = None
        self.fail_on_simulation_failure = True
        self.failure_default_field_values = {}
        self.executable = executable


class _ComputerControllerComponentBase(ABC):

    class InvalidCoreUseValueError(RuntimeError):
        def __init__(self, n_cores):
            message = f"Model must use at least 1 core.\n" + \
                      f"Requested number of cores was {n_cores}"
            super().__init__(message)

    class InvalidTimeLimitSpecified(RuntimeError):
        def __init__(self, *args):
            super().__init__(*args)

    class InvalidQueueIDSpecified(RuntimeError):
        def __init__(self, *args):
            super().__init__(*args)

    class InvalidComputerSpecified(RuntimeError):
        def __init__(self, *args):
            super().__init__(*args)


    def __init__(self, executable, **kwargs):
        self._simulation_information = _DefaultComputeInformation(executable)
        super().__init__(**kwargs)

    def cause_failure_if_simulation_fails(self):
        return self._simulation_information.fail_on_simulation_failure
    
    def on_failure_values(self):
        return self._simulation_information.failure_default_field_values
    
    @property
    def time_limit_seconds(self):
        return self._simulation_information.time_limit_seconds

    @property
    def queue_id(self):
        return self._simulation_information.queue_id

    @property
    def computer(self):
        return self._simulation_information.computer

    @property
    def number_of_cores(self):
        return self._simulation_information.number_of_cores

    @property
    def number_of_local_cores(self):
        if self.computer == local_computer:
            return self.number_of_cores
        else:
            return 1

    @property
    def modules_to_load(self):
        return self._simulation_information.modules_to_load

    @property
    def executable(self):
        return self._simulation_information.executable

    def set_executable(self, executable):
        """
        Set the executable to a user specified executable. This allows for
        changing the executable after initialization for non-MatCal generated
        models, but also allows for changing the executable for 
        MatCal generated models to custom compiled executables.

        :param executable: the name of the executable. It should be in 
            your path or an
            absolute path.
        :type executable: str
        """
        
        check_value_is_nonempty_str(executable, "model executable", 
                                    "model.set_executable")
        self._simulation_information.executable = executable

    def continue_when_simulation_fails(self, **default_field_values):
        """
        Call this method on the model if you want the study to continue when 
        its returns 
        with an error or exit code. By default, It will generate a line of values 
        from -1 to 1 for each required field. The objective will be calculated with 
        these values and the study will continue. If desired custom failure values
        can be bassed in this method call to set the value for those given fields. 
        Each of these fields must have the same length. 

        :param default_field_values: keyword arguments to set the default values
        for failed model evaluations. All arguments passed must have the same 
        number of values. 
        :type default_field_values: list or numpy array

        .. note::
            If the simulation errors out before any data can be returned to 
            MatCal, 
            the entire study will still fail due to not being able to formulate 
            an objective.
        """
        self._simulation_information.fail_on_simulation_failure = False
        self._check_default_values(default_field_values)
        self._simulation_information.failure_default_field_values.update(default_field_values)

    def _check_default_values(self, default_field_values):
        first_length = -1
        for name, value in default_field_values.items():
            for value_i in value:
                if not isinstance(value_i, (int, float)):
                    msg = f"All default falure values must be numeric, passed {type(value_i)}"
                    raise RuntimeError(msg)
            if first_length < 0:
                first_length = len(value)
            else:
                if len(value) != first_length:
                    msg = f"Length of default field {name}, does not have the same length as the first field passed.\n"
                    msg += f"Needed a length of {first_length}, but had a length of {len(value)}"
                    raise RuntimeError(msg)

    def set_number_of_cores(self, n_cores):
        """
        Sets the number of processors for the model to use.

        :param number_cores: The number of processors needed to run a single state
          of the model.
        :type number_cores: int
        """
        if n_cores < 1:
            raise self.InvalidCoreUseValueError(n_cores)
        self._simulation_information.number_of_cores = n_cores

    def run_in_queue(self, queue_id:str, time_limit_hours:float, 
                     is_test: bool=False):
        """ Indicates that this model will be run on a platform requiring 
        queue submission. 
        Will load platform data relevant to the platform the script is 
        launched from. 
        
        :param passed_id: A valid ID for the user.
        :type passed_id: str

        :param time_limit_hours: the number of hours the simulation will be 
            allowed to run.
        :type time_limit_hours: float
        """
        self._set_queue_id(queue_id)
        self.set_time_limit(time_limit_hours)
        if not is_test:
            computer_identifier_func = MatCalComputingPlatformFunctionIdentifier.identify()
            computer = computer_identifier_func()
            self._set_computing_platform(computer)

    def set_time_limit(self, time_limit_hours):
        if isinstance(time_limit_hours, (float, int)) and time_limit_hours > 0:
            self._simulation_information.time_limit_seconds = time_limit_hours*60*60
        else:
            raise self.InvalidTimeLimitSpecified()

    def _set_queue_id(self, passed_id):

        if isinstance(passed_id, str):
            self._simulation_information.queue_id = passed_id
        else:
            raise self.InvalidQueueIDSpecified()

    def add_environment_module(self, module_name):
        """
        Adds an environment module to be loaded before running the model. 
        This must be a valid module on the system
        where the MatCal study is being run. MatCal will run "module load 
        module_name" for the added module before
        running this model. If multiple modules are added, they will be loaded 
        in the order they were added.

        :param module_name: the name of the module to be loaded when running 
            the model.
        :type module_name: str
        """

        check_value_is_nonempty_str(module_name, "module name", 
                                    "model.add_environment_module")

        if self._simulation_information.modules_to_load is None:
            self._simulation_information.modules_to_load = [module_name]
        else:
            self._simulation_information.modules_to_load.append(module_name)
    
    def _set_computing_platform(self, computer):
        """
        Sets the computer for the model to be run on. MatCal has remote computer
        names stored under
        matcal.core.computer_node.CEENode and matcal.core.computer_node.HPCNode. 
        Import either CEENode or HPCNode and
        pass one of their computer names to this function. For example, 
        CEENode.compute10 or HPCNode.uno

        :param computer: The computer to run the model on.
        :type computer: :class:`~matcal.core.computer.computer_node.LocalComputingPlatform` or 
                        :class:`~matcal.core.computer.computer_node.RemoteComputingPlatform`
        """
        if isinstance(computer, (LocalComputingPlatform, RemoteComputingPlatform)):
            self._simulation_information.computer = computer
        else:
            raise self.InvalidComputerSpecified()


class _ResultsInformation:

    def __init__(self):
        self.results_reader_object = FileData
        self.results_filename = "results.csv"
        self.file_type=None

    def read(self, file_path):
        results = self.results_reader_object(file_path, 
                                             file_type=self.file_type)
        return results


class _ResultsRetriever:

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._results_information = _ResultsInformation()


    def set_results_filename(self, filename, file_type=None):
        """
        Sets the name of the model results file.

        :param filename: the filename of the model results output.
        :type filename: str

        :param file_type: the file type of the model results output so the 
            file data importer knows how to read it in.
            Set to "None" by default, so MatCal will try to infer the type 
            from the file extension.
        :type filename: str
        """
        check_value_is_nonempty_str(filename, "filename", "model.set_results_filename")
        self._results_information.results_filename = filename
        if file_type is not None:
           check_value_is_nonempty_str(file_type, "file_type", "model.set_results_filename")
        self._results_information.file_type = file_type

    def _set_results_reader_object(self, results_reader):
        self._results_information.results_reader_object = results_reader


class ModelPreprocessorBase(ABC):
    """
    Base class for model preprocessors not intended for users.
    """
    def __init__(self):
        """"""
    
    @abstractmethod
    def process(self, computing_info, template_dir, **kwargs):
        """"""


class AdditionalFileCopyPreprocessor(ModelPreprocessorBase):
    """
    Model preprocessor not intended for users.
    """

    def process(self, template_dir, additional_files):
        if additional_files:
            logger.info("\n\t\tCopying additional model files to model directory.")
        for other_file in additional_files:
            _copy_file_or_directory_to_target_directory(template_dir, other_file)
            other_file_for_output = os.path.split(other_file)[-1]
            logger.info(f"\t\t\t{other_file_for_output}...")
        logger.info(f"")


class InputFileCopyPreprocessor(ModelPreprocessorBase):
    """
    Model preprocessor not intended for users.
    """
    def process(self, template_dir, input_filename):
        input_name_for_output = os.path.split(input_filename)[-1]
        logger.info(f"\t\tPreparing user supplied input deck \"{input_name_for_output}\"")
        copied_input = _copy_file_or_directory_to_target_directory(template_dir, 
                                                                   input_filename)
        logger.info(f"\t\tInput deck complete.")


def _copy_file_or_directory_to_target_directory(target_dir, source):
    target_filename = os.path.join(target_dir, source.split("/")[-1])
    if not os.path.exists(target_filename):
        if os.path.isdir(source):
            shutil.copytree(source, target_filename)
        else:
            shutil.copyfile(source, target_filename)

    return target_filename


def _get_mesh_template_folder(template_dir):
    if MATCAL_TEMPLATE_DIRECTORY in template_dir:
        template_dir = template_dir.replace(MATCAL_TEMPLATE_DIRECTORY, 
                                            MATCAL_MESH_TEMPLATE_DIRECTORY)
        mesh_files_folder = template_dir
    else:
        mesh_files_folder = os.path.join(MATCAL_MESH_TEMPLATE_DIRECTORY,
                                          template_dir)
    _create_template_folder(mesh_files_folder)
    return mesh_files_folder


def _create_template_folder(template_dir):
    folders = Path(template_dir).parts
    made_folders = []
    for folder in folders:
        made_folders.append(folder)
        made_folders_path = os.path.join(*made_folders)
        if not os.path.exists(made_folders_path):
            os.mkdir(made_folders_path)


class ModelBase(_ResultsRetriever, _ComputerControllerComponentBase):
    """
    Base class for models not intended for users.
    """
    _id_counter = count(0)

    @abstractmethod
    def model_type(self):
        """"""

    @abstractmethod
    def _simulator_class(self):
        """"""

    @abstractmethod
    def _get_simulator_class_inputs(self, state):
        """"""

    @abstractmethod
    def _setup_state(self, state, state_template_dir=None):
        """"""

    def build_simulator(self, state):
        args, kwargs = self._get_simulator_class_inputs(state)
        sim = self._simulator_class(*args, **kwargs)
        return sim

    @property
    def input_file(self):
        return self._input_file

    @property
    def name(self):
        return self._name

    @property
    def results_filename(self):
        return self._results_information.results_filename
        
    @property
    def results_file_type(self):
        return self._results_information.file_type

    def set_name(self, new_name):
        """The name of the model. If the model is run as an external executable,
        the model will be created and run in a directory with this name. 
        If a name is not provided,
        a unique identifier is assigned as the mode name.
        """
        self._name = new_name
        self._revise_derived_names()

    def run(self, state, parameter_collection, target_directory=None):
        """
        Runs the model for the given state and the current value of each 
        parameter in the parameter collection.

        :param state: the state to evaluate the model at.
        :type state: :class:`~matcal.core.state.State`

        :param parameter_collection: the parameter collection that 
            will populate the parameter in the model
        :type parameter_collection: :class:`~matcal.core.parameters.ParameterCollection`

        :param target_directory: target location to be built to run the models in
        :type target_directory: str

        :return: the results from the simulation
        :rtype: :class:`~matcal.core.simulators.SimulatorResults`
        """
        logger.info(f"\tRunning simulation of state \"{state.name}\" for model \"{self.name}\".\n")
        check_item_is_correct_type(state, State, "model.run", "state", )
        check_item_is_correct_type(parameter_collection, 
                                         ParameterCollection, "model.run", 
                                         "parameter_collection")
        self.preprocess(state, target_directory)
        sim = self.build_simulator(state)
        results = sim.run(parameter_collection.get_current_value_dict(), 
                          working_dir=target_directory)
        logger.info(f"\tSimulation of state \"{state.name}\" for model \"{self.name}\" complete.\n")
        return results

    def __init__(self, executable, **kwargs):
        super().__init__(executable=executable, **kwargs)
        self._id_number = next(self._id_counter)
        self.set_name(self.model_type+"_{}".format(self._id_number))
        self._stateless_user_variables = {}
        self._state_user_variables = {}

    def _revise_derived_names(self):
        pass

    def add_constants(self, **kwargs):
        """
        Add additional constant parameters for the model that will be passed 
        to the model before it is evaluated. 
        These key/value pairs will be passed to all states. 
        If these conflict with state variables, these will
        override the state variables. If these conflict with study parameters, 
        the parameter values from the 
        study will take precedent.

        :param kwargs: key/value pair of model constant parameters. 
            For example model.add_simulation_variables(my_var1=5, my_var2=1, etc.)
        :type kwargs: dict(str, str) or dict(str, float)
        """

        for key, value in kwargs.items():
            check_item_is_correct_type(value, (str, Number), "model.add_constants", 
                                       "constant parameter value ")

        self._stateless_user_variables.update(kwargs)

    def add_state_constants(self, state, **kwargs):
        """
        Add additional constant parameters for the model that will be passed to 
        the model before it is evaluated
        for a given state. If the model is not evaluated for this state in the 
        study, these parameters will not be used. If these conflict with 
        experiment state variables, these will
        override the state variables. If these conflict with study parameters, 
        the parameter values from the 
        study will take precedent. Finally, adding a specific state model 
        constant will override general model constants
        added with :meth:`add_constants` method.

        :param state: the state for these parameters
        :type state: :class:`~matcal.core.state.State`

        :param kwargs: key/value pair of model constant parameters. 
            For example model.add_simulation_variables(my_var1=5, my_var2=1, etc.)
        """
        check_item_is_correct_type(state, State, "model.add_state_constants", 
                                   "state")

        for key, value in kwargs.items():
            check_item_is_correct_type(value, (str, Number), "model.add_state_constants", 
                                             "state constant parameter value ")

        if state in self._state_user_variables.keys():
            self._state_user_variables[state].update(kwargs)
        else:
            self._state_user_variables[state] = kwargs

    def reset_constants(self):
        """
        Removes all model constants that were added with the 
        :meth:`add_constants` and :meth:`add_state_constants`
        methods.
        """
        self._stateless_user_variables = {}
        self._state_user_variables = {}
        
    def get_model_constants(self, state=None):
        model_constants = {}
        model_constants.update(self._stateless_user_variables)
        if  (state != None) and (state in self._state_user_variables.keys()):
            model_constants.update(self._state_user_variables[state])
        return model_constants
    
    def _write_state_file(self, state, directory):
        this_state_params_filename = os.path.join(directory, 
                                                  STATE_PARAMETER_FILE)
        dictionary_reporter = MatCalParameterReporterIdentifier.identify()
        model_state_consts = self.get_model_constants(state)
        state_constants = get_parameters_according_to_precedence(state, 
                                                                 model_state_consts)
        dictionary_reporter(this_state_params_filename, 
                            state_constants)
        
    def _make_state_directory(self, state_dir):
        make_clean_dir(state_dir)
        return state_dir

    def _make_template_directory(self, template_dir):
        if template_dir is not None:
            if not os.path.exists(template_dir):
                make_clean_dir(template_dir)
            template_dir = os.path.abspath(template_dir)
            template_dir = os.path.join(template_dir, 
                                        matcal_name_format(self.name))
        else:
            template_dir = os.path.abspath(matcal_name_format(self._name))

        if not os.path.exists(template_dir):
            make_clean_dir(template_dir)

    def get_target_dir_name(self, state, template_dir=None):
        """
        Returns the name of the directory where the model will run if launched
        using :math:`~matcal.core.models.ModelBase.run`.

        rtype: str
        """
        if template_dir is None:
            template_dir = "."

        return os.path.join(template_dir, matcal_name_format(self.name), 
                            matcal_name_format(state.name))

    def confirm_permissions(self):
        if isinstance(self.computer, RemoteComputingPlatform):
            permissions_check = MatCalPermissionsCheckerFunctionIdentifier.identify()
            permissions_check(self.queue_id, self.computer, self.name)

    def preprocess(self, state, target_directory=None):
        """
        Prepares the model for the given state and places the model and 
        associated files in an optional target directory.

        :param state: the state to evaluate the model at.
        :type state: :class:`~matcal.core.state.State`

        :param target_directory: target location to be built to run the models in
        :type target_directory: str

        """
        self._make_template_directory(target_directory)
        logger.info(f"\tPreparing state \"{state.name}\" for model \"{self.name}\" ")
        state_template_dir = self.get_target_dir_name(state, target_directory)
        self._make_state_directory(state_template_dir)
        self._write_state_file(state, state_template_dir)
        self._setup_state(state, state_template_dir)
        logger.info(f"\tState \"{state.name}\" for model \"{self.name}\" initialized\n")

        return state_template_dir


def _python_model_results_reader(filename, file_type=None):
    return matcal_load(filename)


class PythonModel(ModelBase):

    """
    Use a python function as a model to be used in an evaluation set for 
    a MatCal study. The model takes two forms of
    input:

    #. Pass in a locally defined function for the parameter python_function.
    #. Pass in the name of the python function for the parameter 
       python_function as a string and a string which gives
       the full path of the file where the python function is defined. 
       Since MatCal will import from this file,
       it is recommended that nothing is defined or executed 
       in the global names space of that file.

    The python function should take in all parameters being calibrated 
    as input parameters. It should return a
    dictionary with keys being the field names of the responses of interest 
    required by the objectives applied
    to the model and the values for each key should be 1d arrays of the values 
    corresponding to the responses.

    :param python_function: locally defined function or name of function 
        defined in another file.
    :type python function: FunctionType or str

    :param filename: Name of the file where the function is defined if not in 
        the MatCal python input file.
    :type filename: str

    :rtype: dict
    """
    model_type = "python"
    _simulator_class = PythonSimulator
    _input_file = None

    def __init__(self, python_function, filename=None, field_coordinates=None):
        super().__init__(executable="python")
        self._field_coordinates = field_coordinates
        self._function_importer = python_function_importer(python_function, 
                                                           filename)
        self._results_information.results_filename = None
        self._set_results_reader_object(_python_model_results_reader)

    @property
    def python_function(self):
        return self._function_importer.python_function

    def _get_simulator_class_inputs(self, state):
        args = [self.name, self._simulation_information, self._results_information, 
                state, self, self._field_coordinates]
        kwargs = {}

        return args, kwargs

    def _setup_state(self, state, template_dir=None):
        """""" 
    

class MatCalSurrogateModel(PythonModel):
    model_type = "matcal_surrogate"
    
    """
    A Model class that creates the correct interface between MatCal surrogates 
    and the environment necessary for models within MatCal. This model class 
    lightly extends the :class:`~matcal.core.models.PythonModel` class. 
    """
    
    
    def __init__(self, surrogate:MatCalSurrogateBase):
        """
        Generates a MatCal model from an instantiated MatCal surrogate model
        
        :param surrogate: The MatCal surrogate to use as the base for a model
        :type surrogate: :class:`~matcal.core.surrogate.MatCalSurrogateBase`
        """
        wrapped_surrogate = _MatCalSurrogateWrapper(surrogate)               
        super().__init__(wrapped_surrogate)


class UserExecutableModel(ModelBase):

    """
    Run any executable accessible in the path as a model to be used in 
    an evaluation set for 
    a MatCal study. 

    If any files are needed in order for the model to run, they can be added using
    :meth:`~matcal.core.models.UserExecutableModel.add_necessary_files`. These 
    files will be copied to a templates folder and then linked into the working directory
    where the model executable is launched for each evaluation.

    :param executable: The executable to be run. This can be an executable found 
        in path or the full path 
        to an executable should be provided.
    :type executable: str

    :param arguments: Arguments to the executable that are required for 
        the executable to run 
        correctly. This can be a comma separated list or unpacked list of strings. 
    :type arguments: str

    :param results_filename: The filename where the model results will be stored. 
    :type results_filename: str


    :param results_file_type: The file type for the results file. This should be a valid 
        file type for the :func:`~matcal.core.data_importer.FileData` function. 
    :type results_filename: str

    """
    model_type = "user_executable_model"
    _simulator_class = ExecutableSimulator
    _input_file = None

    def __init__(self, executable, *arguments, results_filename=None, results_file_type=None):
        check_value_is_nonempty_str(executable, "executable", "UserExecutableModel")
        super().__init__(executable=executable)
        for arg in arguments:
            check_value_is_nonempty_str(arg, "arguments", "UserExecutableModel")
        self.set_results_filename(filename=results_filename, file_type=results_file_type)
        self._arguments = list(arguments)
        self._additional_sources_to_copy = []
        
    def _get_simulator_class_inputs(self, state):
        args = [self.name, self._simulation_information, 
                self._results_information, state, self.get_model_constants(state),
                ]
        kwargs = {"commands":[self.executable]+self._arguments}

        return args, kwargs

    def _setup_state(self, state, state_template_dir=None):
        additional_file_copier = AdditionalFileCopyPreprocessor()
        additional_file_copier.process(state_template_dir, self._additional_sources_to_copy)
        
    def add_necessary_files(self, *needed_files):
        """
        ":param needed_files: additional files or directories that need to be
            in the working directory of the model so that it can
            run. These are include files that the main input file may need or 
            mesh and other data files.
        :type needed_files: list(str)
        """
        other_sources = _convert_list_of_files_to_abs_path_list(needed_files)
        self._additional_sources_to_copy = other_sources