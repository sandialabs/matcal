"""
The study base module contains the base class for all MatCal 
studies. It is not user facing but must be used as the base class 
for any new studies. 
"""
from abc import ABC, abstractmethod
from collections import OrderedDict
from datetime import datetime
from getpass import getuser
import glob
from inspect import isclass
import numpy as np
import os
import shutil
from sys import argv

from matcal.core.constants import (FINAL_RESULTS_FILENAME, IN_PROGRESS_RESULTS_FILENAME, 
                                   MATCAL_WORKDIR_STR, STATE_PARAMETER_FILE, 
                                   MATCAL_TEMPLATE_DIRECTORY, MATCAL_MESH_TEMPLATE_DIRECTORY)
from matcal.core.data import (DataCollection, MaxAbsDataConditioner,
                               DataConditionerBase)
from matcal.core.evaluation_set import StudyEvaluationSet
from matcal.core.logger import initialize_matcal_logger
from matcal.core.models import ModelBase
from matcal.core.objective import (ObjectiveCollection, ObjectiveSet, 
                                   SimulationResultsSynchronizer)
from matcal.core.parameters import (Parameter, ParameterCollection, 
                                    UserDefinedParameterPreprocessor)
from matcal.core.parameter_batch_evaluator import (ParameterBatchEvaluator, 
                                                   flatten_evaluation_batch_results) 
from matcal.core.plotting import _NullPlotter, StandardAutoPlotter
from matcal.core.pruner import DirectoryPrunerKeepAll, DirectoryPrunerBase, Eliminator
from matcal.core.serializer_wrapper import matcal_save
from matcal.core.state import StateCollection, SolitaryState
from matcal.core.utilities import (_sort_workdirs, check_item_is_correct_type, 
                                   check_value_is_positive_integer) 
from matcal.version import __version__

logger = initialize_matcal_logger(__name__)

class StudyBase(ABC):
    """
    Base class for all MatCal Studies, not intended for users.
    """

    @property
    @abstractmethod
    def study_class(self):
        """"""

    @abstractmethod
    def _run_study(self):
        """"""

    @abstractmethod
    def _format_parameters(self, params):
        """"""

    @abstractmethod
    def _format_parameter_batch_eval_results(self, batch_raw_objectives, 
                                             flattened_batch_results, 
                                             total_objs, parameter_sets, batch_qois):
        """"""

    @abstractmethod
    def _study_specific_postprocessing(self):
        """"""

    @abstractmethod
    def restart():
        """"""

    @property
    @abstractmethod
    def _needs_residuals(self):
        """"""

    class StudyTypeError(Exception):
        def __init__(self, *args):
            super().__init__(*args)

    class StudyError(Exception):
        def __init__(self, *args):
            super().__init__(*args)

    class StudyInputError(Exception):
        def __init__(self, *args):
            super().__init__(*args)

    _template_directory_name = "matcal_template"
    _state_filename = STATE_PARAMETER_FILE

    def __init__(self, *parameters):
        """
        :param parameters: The parameters of interest for the study.
        :type parameters: list(:class:`~matcal.core.parameters.Parameter`) or
            :class:`~matcal.core.parameters.ParameterCollection`

        :raises StudyTypeError: if parameters is of incorrect type.
        """
        self._initialize_log_file()

        self._evaluation_sets = OrderedDict()
        self._parameter_preprocessors = []
        self._parameter_batch_evaluator = None
        self.set_parameters(*parameters)
        self._next_evaluation_id_number = 1

        self._name = self.study_class
        self._total_cores_available = 1
        self._results = None
        self._results_reporting = None
        self.set_results_storage_options()
        self._plotter = _NullPlotter()

        self._restart = False
        self._perform_data_purge = True

        self._assessor = DirectoryPrunerKeepAll()

        self._use_threads = False
        self._always_use_threads = False
        self._run_async = True
    
        self._working_directory = None
        self._remove_existing_working_directory = False
        self._initial_directory = os.getcwd()

        self._final_results_filename = None

        self._generated_collection_id = 0

    def _initialize_log_file(self):
        logger.info("Running MatCal")
        logger.info("Started by: {}".format(getuser()))
        logger.info("Started on: {}".format(datetime.today().strftime('%m/%d/%Y at %I:%M:%S %p')))
        logger.info("MatCal Version: {}\n".format(__version__))
        self._add_file_contents_to_log(self._get_input_file_path())

    def _add_file_contents_to_log(self, filepath):
        if filepath:
            logger.info("User input script:")
            if filepath == "html":
                #Need to skip this part when building documentation
                pass
            else:
                with open(filepath, 'r') as file:
                    for line_number, line in enumerate(file.readlines()):
                        logger.info("\t{:4d}>>> {}".format(line_number, line.strip('\n')))
        logger.info("End user input script!\n")

    @property
    def results(self):
        """
        Return access to the study's results. Will return None, if study has not
        been run. 
        """
        return self._results
        
    def add_evaluation_set(self, model, objectives, data=None, states=None, 
                           data_conditioner_class=MaxAbsDataConditioner):
        """
        Adds an evaluation set to the study. An evaluation set is a set of datasets, 
        objectives and states that
        are applicable to a model. For each evaluation set, the model will be evaluated 
        for every state in the set.
        The results from each model state will be compared to each dataset its state.
        This comparison consists of each objective in the passed objectives.

        :param model: The model that will generate results for comparison to the data in the set.
        :type model:  valid model type from the  :mod:`~matcal.core.models` module

        :param objectives: The objectives to quantitatively compare the model results to the data.
        :type objectives: :class:`~matcal.core.objective.Objective` or
            :class:`~matcal.core.objective.ObjectiveCollection`

        :param data: The data to be evaluated with this evaluation set. Data is not required
            when this method is called with a 
            :class:`~matcal.core.objective.SimulationResultsSynchronizer`.
        :type data: :class:`~matcal.core.data.Data` or :class:`~matcal.core.data.DataCollection`

        :param states: A subset of states in the data that are of interest for this study.
        :type states: :class:`~matcal.core.state.State` or 
            :class:`~matcal.core.state.StateCollection`

        :param data_conditioner_class: the class that will be used as a data conditioner for 
            this evaluation set. See :mod:`~matcal.core.data` for valid data conditioners.
        :type data_conditioner: class

        :raises StudyTypeError: if passed arguments are of the incorrect type.
        :raises StudyError: if all the passed states are not in the data.
        """
        self._check_item_is_correct_type(model, ModelBase, "model")
        objective_collection = self._singleton_to_collection(objectives, ObjectiveCollection, 
                                                             "objectives")
        self._check_data_and_objectives(objective_collection, data)

        state_collection, exp_data_col = self._get_states_and_data(states, 
                                                                   data,
                                                                   objective_collection)
        self._check_is_valid_data_conditioner_class(data_conditioner_class)
        self._update_evaluation_sets(exp_data_col, model, objective_collection, 
                                     state_collection, data_conditioner_class)

    def set_results_storage_options(self, data:bool=True, qois:bool=True,
                                    residuals:bool=True, objectives:bool=True, 
                                    weighted_conditioned:bool=False, 
                                    results_save_frequency:int=1):
        """
        Set which history information to save and return with the study results. 
        You can also down sample which evaluations to save using results_save_frequency.
        This is particularly useful if you wish to not store finite difference evaluations 
        for gradient based studies.
        The total objective is always stored.
       
        :param data: Store the raw data for each simulation and the raw experimental 
            data for each objective for each desired evaluation.
        :type data: bool

        :param qois: Store the QoIs for each objective for each desired evaluation.
            This includes both experiment and simulation QoIs
        :type qois: bool
                
        :param residuals: Store the residuals for each objective for each desired 
            evaluation.
        :type residuals: bool

        :param objectives: Store the objective by state and evaluation set for each desired 
            evaluation.
        :type objectives: bool

        :param weighted_conditioned: Store the weighted and conditioned values 
            for each desired evaluation. This will save the weighted and conditioned, 
            residuals, simulation qois and experiment qois.
        :type weighted_conditioned: bool

        :param results_save_frequency: Set how the results save interval.
            For studies where finite difference derivatives are used, an interval 
            of :math:`n+1` will exclude  finite difference results from the saved
            results history.
        :type results_save_frequency: int

        """
        check_item_is_correct_type(data, bool, "study.set_results_storage_options", 
                                   "data")
        check_item_is_correct_type(qois, bool, "study.set_results_storage_options", 
                                   "qois") 
        check_item_is_correct_type(residuals, bool, "study.set_results_storage_options", 
                                   "residuals")
        check_item_is_correct_type(objectives, bool, "study.set_results_storage_options", 
                                   "objectve")
        check_item_is_correct_type(weighted_conditioned, bool, "study.set_results_storage_options", 
                                "weighted_conditioned")
        check_value_is_positive_integer(results_save_frequency, "results_save_frequency",
                                        "study.set_results_storage_options")
        self._results_reporting = OrderedDict()
        self._results_reporting["record_data"] = data
        self._results_reporting["record_qois"] = qois
        self._results_reporting["record_residuals"] = residuals
        self._results_reporting["record_objectives"] = objectives
        self._results_reporting["record_weighted_conditioned"] = weighted_conditioned
        self._results_reporting["results_save_frequency"] = results_save_frequency

    def _check_data_and_objectives(self, objective_collection, data):
        if data is None:
            for objective in objective_collection.values():
                if not isinstance(objective, SimulationResultsSynchronizer):
                    raise ValueError("Data is required for an evaluation set"
                                     " when not using a single \"SimulationResultsSynchronizer\""
                                     f" objective. A \"{type(objective)}\" was" 
                                     " added to the evaluation set so data is required.")
                else:
                    if len(objective_collection.values()) > 1:
                        raise ValueError("If using a \"SimulationResultsSynchronizer\", "
                                         " only a single \"SimulationResultsSynchronizer\"" 
                                         " objective can be added per evaluation set. No other"
                                         " objectives or objective types can be added with it.")

    def _check_is_valid_data_conditioner_class(self, data_conditioner_class):
        is_valid_conditioner = False
        if isclass(data_conditioner_class):
            is_valid_conditioner = issubclass(data_conditioner_class, DataConditionerBase)
        if not is_valid_conditioner:
            raise TypeError("The data conditioner class must be a class derived "
                            "from the DataConditionerBase. Invalid \"data_conditioner_class\" "
                            "passed to \"study.add_evaluation_set\"")

    def plot_progress(self):
        """
        Calling this method will cause matcal to generate automatic plots after 
        each batch of parameter evaluations. These plots are made using the 
        standard plotter and will show things such as objective value evolution. 
        """
        self._plotter = StandardAutoPlotter()

    def _check_item_is_correct_type(self, item, desired_type, message):
        if not isinstance(item, desired_type):
            raise self.StudyTypeError(
                f"The study expected a {desired_type} type "
                f"for the added {message}, but received object of type {type(item)}")

    @property
    def final_results_filename(self):
        """
        Returns the filename for the final results file for the current study.

        return: final results filename as an absolute path
        rtype: str
        """
        return self._final_results_filename

    def _get_states_and_data(self, states, data, objectives):
        if data is not None:
            exp_data_col = self._singleton_to_collection(data,
                                                         DataCollection, "data")
            state_collection = self._determine_eval_set_states(exp_data_col, 
                                                               states)
        else:
            state_collection, exp_data_col = self._generate_eval_set_data(objectives,
                                                                           states)
        return state_collection, exp_data_col

    def _determine_eval_set_states(self, exp_data_col, states):
        if states is None:
            state_collection = exp_data_col.states
        else:
            state_collection = self._singleton_to_collection(states, StateCollection, "states")
            self._check_states_valid_subset_of_data(states, exp_data_col)
        return state_collection
    
    def _generate_eval_set_data(self, objectives, states):
        if states is None:
            states = SolitaryState()
        state_collection = self._singleton_to_collection(states, StateCollection, "states")
        experimental_data_collection = DataCollection(f"Sim results synchronizer" 
                                                           " generated"
                                                           f" {self._generated_collection_id}")
        self._generated_collection_id +=1
        for state in state_collection.values():
            experimental_data_collection.add(self._generate_data_from_objectives(objectives,
                                                                                    state))
        return state_collection, experimental_data_collection

    def _generate_data_from_objectives(self, objectives, state):
        return list(objectives.values())[0]._generate_experimental_data_qois(state)

    def _check_states_valid_subset_of_data(self, states, data_collection):
        data_states = list(data_collection.states.values())
        for state in states.values():
            if state not in data_states:
                raise self.StudyError(f"The state \"{state.name}\" in state "
                                      f"collection \"{states.name}\" does "
                                      "not exist in the data "
                                      f"collection \"{data_collection.name}\". "
                                      "They were passed to the "
                                      "study as an evaluation set and all "
                                      "states in the state collection must exist "
                                      "in the data collection. Check "
                                      "input.")

    def _update_evaluation_sets(self, experimental_data_collection, model, objective_collection,
                                state_collection, data_conditioner_class):
        new_objective_set = self._create_new_objective_set(experimental_data_collection, 
                                                           objective_collection,
                                                           state_collection, data_conditioner_class)
        if model in self._evaluation_sets.keys():
            self._check_for_repeated_objective_set_components(model, new_objective_set)
            self._evaluation_sets[model].add_objective_set(new_objective_set)
        else:
            self._check_for_repeated_model_name(model)
            self._evaluation_sets[model] = StudyEvaluationSet(model, new_objective_set)

    def _create_new_objective_set(self, experimental_data_collection, objective_collection, 
                                  state_collection, data_conditioner_class):
        new_objective_set = ObjectiveSet(objective_collection, experimental_data_collection, 
                                         state_collection, data_conditioner_class)
        return new_objective_set

    def _check_for_repeated_objective_set_components(self, model, new_objective_set):
        for obj_set in self._evaluation_sets[model].objective_sets:
            if (obj_set.data_collection == new_objective_set.data_collection and
                obj_set.states == new_objective_set.states and 
                obj_set.objectives == new_objective_set.objectives):

                raise self.StudyError("A repeated evaluation set was added for model \'{}\'."
                                      " Check input".format(model.name))

    def _check_for_repeated_model_name(self, new_model):
        if new_model.name in self._get_model_names():
            error_str = f"A model named \"{new_model.name}\" has already been added to the study."
            error_str += f"\nThe model being added with same name is:\n{new_model}"
            error_str += f"\nThe model already part of the study with this "
            error_str += f"name is:\n{self._get_model_by_name(new_model.name)}"
            raise self.StudyInputError(error_str)

    def _get_model_names(self):
        model_names = []
        for model in self._evaluation_sets.keys():
            model_names.append(model.name)
        return model_names

    def _get_model_by_name(self, model_name):
        for model in self._evaluation_sets.keys():
            if model.name == model_name:
                return model
        return None

    def _get_input_file_path(self, argv=argv):
        is_unit_test = False
        is_doc_build = False
        for arg in argv:
            if 'unittest' in str(arg): 
                is_unit_test = True
            if "html" in str(arg):
                is_doc_build = True
                #skip for doc build
                pass 
        if is_unit_test or is_doc_build:
            return None
        else:
            if len(argv) > 1:
                input_index = 0
                for index, arg in enumerate(argv):
                    if arg == '-i' or arg == '--input':
                        input_index = index + 1
                return os.path.abspath(argv[input_index])
            else:
                return None

    def launch(self):
        """
        This launches the study. Note that at least one evaluation set must be added with
        :meth:`~add_evaluation_set`.

        :return: study specific results.

        :raises StudyError: if no evaluation sets have been added.
        """
        self._initialize_study_and_batch_evaluator()
        
        self._purge_unused_data()
        logger.info("Launching study...\n")
        
        self._results = self._run_study()
        logger.info("Study complete!\n")

        self._study_specific_postprocessing()
        logger.enable_traceback()
        self._purge_unneeded_matcal_information()
        self._export_final_results()
        self._return_to_initial_directory()
        
        date_time_str = datetime.today().strftime('%m/%d/%Y at %I:%M:%S %p')
        logger.info(f"MatCal completed at: {date_time_str}")
        return self._results

    def _export_final_results(self):
        self._set_final_results_name()
        self._results.save(self._final_results_filename)

    def _set_final_results_name(self):
        filename = f"{FINAL_RESULTS_FILENAME}.joblib"
        if self._working_directory is not None:
            filename = os.path.join(self._working_directory, filename)
        self._final_results_filename = filename

    def _go_to_working_directory(self):
        if self._working_directory is not None:
            if (os.path.exists(self._working_directory) and not self._restart 
                and not self._remove_existing_working_directory):
                raise FileExistsError(f"The working directory \"{self._working_directory}\"" 
                               " already exists. Rename it or remove it before running"
                               " MatCal. Or pass \"remove_existing=True\" to "
                               "the \".set_working_directory\" method.")
            elif not os.path.exists(self._working_directory):
                os.mkdir(self._working_directory)
            os.chdir(self._working_directory)
            
    def _purge_unused_data(self):
        if self._perform_data_purge:
            for eval_set in self._evaluation_sets.values():
                for objective_set in eval_set.objective_sets:
                    objective_set.purge_unused_data()

    def _initialize_study_and_batch_evaluator(self):
        logger.info("Initializing study...")
        self._check_if_repeat_launch()
        self._check_restart()
        self._check_that_evaluation_sets_is_populated()
        if self._restart:
            logger.info("Resuming study from restart information")
            self._go_to_working_directory()
        else:
            self._go_to_working_directory()
            self._delete_old_study_files()
        self._initialize_evaluation_sets()
        parameter_batch_evaluator = ParameterBatchEvaluator(self._total_cores_available, 
                                                            self._evaluation_sets,
                                                            self._use_threads, 
                                                            self._always_use_threads,
                                                            self._run_async)
        self._parameter_batch_evaluator = parameter_batch_evaluator
        self._initialize_results()
        logger.info("Study initialized!\n")
        return parameter_batch_evaluator

    def _check_if_repeat_launch(self):
        if self._results != None:
            raise self.RepeatLaunchError()

    class RepeatLaunchError(RuntimeError):

        def __init__(self):
            message = "Study instance is not clean, likely due to being rerun in same python session."
            message += "\nTo solve this start a new python session or "
            message += "create a new study instance for each call of 'launch'."
            super().__init__(message)

    def _check_restart(self):
        if self._use_threads and self._restart:
            raise RuntimeError("Use of Threads and Restart functionality currently does not work." \
            "Please do not invoke 'set_use_threads' with restarts.")

    def _initialize_results(self):
        if self._results is None:
            self._results = StudyResults(**self._results_reporting)

    def _check_that_evaluation_sets_is_populated(self):
        if not self._evaluation_sets:
            raise self.StudyError("Add evaluation sets to the study before launch."
                                  " Use the add_evaluation_set method.")

    def _initialize_evaluation_sets(self):
        for eval_set in self._evaluation_sets.values():
            eval_set.model.confirm_permissions()
            eval_set.prepare_model_and_simulators(MATCAL_TEMPLATE_DIRECTORY, 
                                                  self._restart)

    def set_core_limit(self, core_limit, override_max_limit=False):
        """
        Sets the total number of cores that the study may use.

        :param core_limit: The max number of cores that the study can use at any time.
        :type core_limit: int

        :param override_max_limit: Override the default max cores that can be specified
          for a given study. The current limit
          of 500 is recommended by the MatCal team but might not be best for all cases.

        :raises StudyTypeError: if the passed value is not an int.
        """
        self._check_item_is_correct_type(core_limit, int, "core limit")
        max_study_core_limit=500
        if core_limit > max_study_core_limit and not override_max_limit:
            raise self.StudyInputError(f"It is recommended to not allow more than "
                                       f"{max_study_core_limit} cores to be used in a study."
                                        " If you want to use more than this, pass "
                                        "\"True\" to the override_max_limit variable.")
        self._total_cores_available = core_limit

    def set_cleanup_mode(self, new_pruner: DirectoryPrunerBase):
        '''
        Changes the pruner to the object passed as an argument
        '''
        self._assessor = new_pruner

    def _singleton_to_collection(self, item, collection_class, type_name):
        collection_type = collection_class.get_collection_type()
        if isinstance(item, collection_type):
            new_col_name = f"Study generated collection {self._generated_collection_id}"
            new_col = collection_class(new_col_name, item)
            self._generated_collection_id += 1
            return new_col
        elif isinstance(item, collection_class):
            self._check_collection_is_not_empty(collection_type, item, type_name)
            return item
        else:
            raise self.StudyTypeError("The incorrect type has been passed to study for its {}. "
                                      "Expected object of type {} or {}, "
                                      "but received object of type{}.".format(type_name, 
                                                                              collection_class,
                                                                              collection_type,
                                                                              type(item)))
        
    def _check_set_params_arguments(self, item_tuple, collection_class, type_name):
        if len(item_tuple) == 1:
            item = item_tuple[0]
            if isinstance(item, ParameterCollection):
                return item
            elif isinstance(item, Parameter):
                return ParameterCollection(f'{self.study_class}_parameters', item)
            else:
                raise TypeError("Study.set_parameters requires inputs to be a "
                                "Parameter or ParameterCollection classes. Received "
                                f" item of type {type(item)} ")
        else:
            return ParameterCollection(f'{self.study_class}_parameters', *item_tuple)

    def _check_collection_is_not_empty(self, collection_type, item, type_name):
        if len(item) == 0:
            raise self.StudyError(f"The {type_name} passed to study is empty. "
                                  f"It must contain at least one item of type {collection_type}. "
                                  "Check input.")

    def _purge_unneeded_matcal_information(self):
        eliminator = Eliminator()
        eliminator.eliminate(self._assessor.assess())

    def _return_to_initial_directory(self):
        current_directory = os.path.abspath(os.getcwd())
        if current_directory != self._initial_directory and self._initial_directory is not None:    
            os.chdir(self._initial_directory)
        
    def _delete_old_study_files(self):
        logger.info("\n\tRemoving old study files...")
        for old_dir in glob.glob(MATCAL_WORKDIR_STR+".*"):
            shutil.rmtree(old_dir)
        for old_dir in glob.glob(MATCAL_TEMPLATE_DIRECTORY):
            shutil.rmtree(old_dir)
        for old_dir in glob.glob(MATCAL_MESH_TEMPLATE_DIRECTORY):
            shutil.rmtree(old_dir)
        logger.info("\tOld study files removed!\n")

    def add_parameter_preprocessor(self, parameter_preprocessor):
        """
        Add a parameter preprocessor to the study that will operate on the parameters 
        before they are sent to the models. 
        See :class:`~matcal.core.parameters.UserDefinedParameterPreprocessor`.

        :param parameter_preprocessor: the parameter preprocessor that will modify and update 
            the given model parameters 
        :type parameter_preprocessor:
            :class:`~matcal.core.parameters.UserDefinedParameterPreprocessor`
        """
        self._check_item_is_correct_type(parameter_preprocessor, 
                                         UserDefinedParameterPreprocessor, "parameter "
                                        "preprocessor")
        self._parameter_preprocessors.append(parameter_preprocessor)
        
    def set_parameters(self, *parameters):
        """
        :param parameters: The parameters of interest for the study.
        :type parameters: :class:`~matcal.core.parameters.Parameter` or
            :class:`~matcal.core.parameters.ParameterCollection`

        :raises StudyTypeError: if the parameters are of incorrect type.
        """
        param_collection = self._check_set_params_arguments(parameters, ParameterCollection, 
                                                            "parameters")
        param_collection = self._apply_parameter_preprocessing(param_collection)
        self._parameter_collection = param_collection 

    def _apply_parameter_preprocessing(self, 
                                       param_col:ParameterCollection)->ParameterCollection:
        return param_col

    def set_use_threads(self, always_use_threads=False):
        """
        By default, MatCal assumes that the model being run is CPU intensive. 
        As a result, it runs each model in a subprocess which can result in some 
        additional overhead. If running studies cheaper python models, it may be beneficial
        to use threading instead of a subprocess. Using this method will run the study
        with threading if only one model can be evaluated at a time. You can optionally
        run with threads even with concurrent model evaluations with the \"always_use_threads\"
        option; however, this can be 
        less reliable. For large memory calibrations, we always recommend using subprocess.  

        Finally, any external executable is always run using subprocess, but threading 
        can be use to manage that job and return its results.

        :param always_use_threads: if true, MatCal will use threads over
            subprocess for concurrent modeling jobs. 
            Defaults to False.
        :type always_use_threads: bool
            
        """
        self._check_item_is_correct_type(always_use_threads, bool, "always_use_threads")
        self._always_use_threads = always_use_threads
        self._use_threads = True

    def run_in_serial(self):
        """
        Tell MatCal to run evaluations in serial. This is only recommended if the study is 
        serial, like a MCMC Bayes Study, and the model evaluations are fast, like a python
        model. 
        
        Running in serial avoids the overhead of reloading large data sets that are necessary 
        in async studies. 
        """
        self._run_async = False
    
    def set_working_directory(self, working_directory, remove_existing=False):
        """
        By default, MatCal runs in the current working directory. This
        method allows the user to specify a subdirectory in the current directory
        for the study to be run in. This method will create only the 
        last directory in the path. So if the desired subdirectory is under 
        a multiple folders from the current directory MatCal will error 
        if the head of the path does not exist. See :meth:`os.path.split` for a 
        definition of the path \"head\". 

        :param working_directory: The desired working directory for the current study. 
            MatCal will only create the last folder if the path is a nested path.
        :type working_directory: str

        :param remove_existing: If True, then the directory will be removed if 
            pre-existing at study launch.
        :type remove_exiting: bool
        """
        self._check_item_is_correct_type(working_directory, str, "study working directory")
        check_item_is_correct_type(remove_existing, bool, "study.set_working_directory", 
                                   "remove_existing")
        head, tail = os.path.split(working_directory)
        if head != "" and \
            not os.path.exists(head):
            raise self.StudyInputError("Specified working directory is invalid. \nMatCal will only "
                                  "create the last directory in a specified path. \nCreate "
                                  "the parent directories or change the desired working directory.")
        self._working_directory = os.path.abspath(working_directory)
        self._initial_directory = os.getcwd()
        self._remove_existing_working_directory = remove_existing

    def _matcal_evaluate_parameter_sets_batch(self, parameter_sets, is_finite_difference_eval=False, is_restart=False):
        formatted_parameter_sets = self._prepare_parameter_sets_to_evaluate(parameter_sets)
        evaluator_func = self._parameter_batch_evaluator.evaluate_parameter_batch
        batch_results = evaluator_func(formatted_parameter_sets, 
                                       self._needs_residuals, is_restart)
        batch_raw_objectives, total_objectives, batch_qois = _unpack_evaluation(batch_results)
        
        _record_results(self._results, formatted_parameter_sets, batch_raw_objectives, 
                             total_objectives, batch_qois, is_finite_difference_eval)
        self._plotter.plot()
        
        flattened_batch_results = flatten_evaluation_batch_results(batch_raw_objectives)
        return self._format_parameter_batch_eval_results(batch_raw_objectives, 
                                                         flattened_batch_results,
                                                         total_objectives, 
                                                         parameter_sets, batch_qois)

    def _prepare_parameter_sets_to_evaluate(self, parameter_sets):
        prepared_parameter_sets = OrderedDict()
        if not isinstance(parameter_sets, list):
            parameter_sets = [parameter_sets]
        for parameter_set in parameter_sets:
            eval_dir_name, param_dict  = self._get_eval_dir_and_parameter_dict(parameter_set)
            prepared_parameter_sets[eval_dir_name] = param_dict

        return prepared_parameter_sets

    def _get_eval_dir_and_parameter_dict(self, param_set):
        eval_dir_name = MATCAL_WORKDIR_STR+f".{self._next_evaluation_id_number}"
        param_dict = self._format_parameters(param_set)
        processed_param_dict = self._preprocess_parameters(param_dict)
        self._next_evaluation_id_number += 1
        return eval_dir_name, processed_param_dict
    
    def _updated_next_evaluation_id_number(self):
        if self._restart:
            self._next_evaluation_id_number = self._results.number_of_evaluations + 1

    def _preprocess_parameters(self, params):
        processed_params = OrderedDict(**params)
        for param_preprocessor in self._parameter_preprocessors:
            processed_params.update(param_preprocessor(processed_params))
        return processed_params


class StudyResults:
    """
    A class used to store the results of a study and facilitate user 
    processing of results.

    To get study specific results in a dictionary access them 
    with :meth:`~matcal.core.study_base.StudyResults.outcome`.
    Or access them as attributes on the class it self. 

    For example, if you store a calibration result as a variable with name "cal_results", 
    you can access the outcome as follows:

    .. code-block:: python

        best_params = cal_results.outcome["best"]
        best_y = cal_results.outcome["best:y"]
        best_params = cal_results.best.to_dict()
        best_y = cal_results.best.y
    
    The dictionary "best_params" and variable "best_y" is the same 
    using both methods to access study results.
    """
    def __init__(self, record_qois:bool=True, record_residuals:bool=True,
                 record_objectives:bool=True, record_data:bool=True,
                 record_weighted_conditioned:bool=False, results_save_frequency:int=1, 
                  **kwargs):
        """
        Users should not need to initialize this class.

        :param record_qois: Indicates if QoI history should be recorded (default is True)
        :type record_qoi_hist: bool

        :param record_residuals: Indicates if residual history 
            should be recorded.
        :type record_residual_hist: bool
        
        :param record_objectives: Indicates if the objective 
            history for individual objectives should be recorded.
        :type record_objective_hist: bool
        
        :param record_data: Indicates if the raw simulation results history 
            and raw experimental data
            should be recorded.
        :type record_data_hist: bool
        
        :param record_weighted_conditioned: Indicates if the weighted and conditioned
            versions of the QoIs are stored. 
        :type record_intermediates_hist: bool
        
        :param results_save_frequency: The frequency at which results should be saved.
        :type results_save_frequency: int
        """
        self._record_data = record_data
        self._record_qois = record_qois
        self._record_residuals = record_residuals
        self._record_weighted_conditioned = record_weighted_conditioned
        self._record_objectives = record_objectives
        self._save_freq = results_save_frequency
        self._parameter_history = OrderedDict()
        self._evaluation_sets = []
        self._evaluation_ids = []
        self._qoi_history = None
        self._simulation_history = OrderedDict()
        self._obj_history = None
        self._total_objective_history = []
        self._outcome = None
        self._number_of_evaluations=0
        
    @property
    def should_record_parameters(self):
        should_record = (self._record_objectives or self._record_qois or 
                        self._record_residuals or self._record_data or
                        self._record_weighted_conditioned)
        return should_record
    @property
    def outcome(self)->dict:
        """
        Stores the relevant outcomes of the study performed

        Depending upon the type of study, the returned dictionary will have 
        differently named keys for each parameter. For example, a calibration 
        will return a dictionary with keys named 'best:<parameter_name>' for each 
        parameter, while a sensitivity study will have keys titled 'sobol:<parameter_name>'.

        All of the study specific results can also be accessed as attributes
        or nested attributes of the :class:`~matcal.core.study_base.StudyResults`
        object. 

        :return: The outcomes of the study
        :rtype: dict
        :raises RuntimeError: If no outcome is defined
        """
        if self._outcome is None:
            raise RuntimeError("No outcome defined")
        
        return self._outcome
        
    @property
    def parameter_history(self):
        """
        Stores the history of parameters evaluated during the study.

        :return: The history of parameters evaluated during the study
            where keys are the parameter names and the values are a list 
            which include all parameter values in the order they were evaluated.
        :rtype: OrderedDict (str, list(float))
        """

        return self._parameter_history

    @property
    def simulation_history(self):
        """
        Stores the history of simulations performed during the study

        :return: The history of simulations performed during the study
            where keys are the model names and values a 
            :class:`~matcal.core.data.DataCollection`
            of simulation results for each model. The repeats in the 
            :class:`~matcal.core.data.DataCollection` for a given model 
            and state are the simulation results for that model and state 
            in the order of evaluation for the study.
        :rtype: OrderedDict(str, :class:`~matcal.core.data.DataCollection`)
        """
        return self._simulation_history
        
    @property
    def number_of_evaluations(self):
        """
        The number of evaluations performed in the study

        :return: The number of evaluations performed in the study
        :rtype: int
        """
        return self._number_of_evaluations
        
    @property
    def exit_status(self):
        """
        Stores any termination information from the study

        :return: The termination information from the study
        :rtype: str
        """
        return self._exit_message
 
    @property
    def evaluation_sets(self):
        """
        Stores the names of evaluation set results 
        stored in the results. These names are needed to access results 
        for certain result types such as QoIs from models and experiments.

        :return: The evaluation set names stored in the results
        :rtype: list(str)
        """
        return self._evaluation_sets
        
    @property
    def objective_history(self):
        """
        Stores the history of objectives evaluated during the study. 
        The keys are the evaluation set names stored in 
        :meth:`~matcal.core.study_base.StudyResults.evaluation_sets`.
        Each value contains a populated
        :class:`~matcal.core.study_base.ObjectiveInformation` object
        corresponding to the evaluation set name key.

        :return: The history of objectives evaluated during the study
        :rtype: OrderedDict(str, :class:`~matcal.core.study_base.ObjectiveInformation`)
        """
        return self._obj_history

    @property
    def qoi_history(self):
        """
        Stores the history of QoI for the objectives evaluated during the study. 
        The keys are the evaluation set names stored in 
        :meth:`~matcal.core.study_base.StudyResults.evaluation_sets`.
        Each value contains a populated
        :class:`~matcal.core.study_base.QoiInformation` object
        corresponding to the evaluation set name key.

        :return: The history of QoIs for each evaluation during the study
        :rtype: OrderedDict(str,  :class:`~matcal.core.study_base.QoiInformation`)
        """
        return self._qoi_history

    @property
    def total_objective_history(self):
        """
        The history of the overall total objective 
        evaluated during the study in order of evaluation.

        :return: The overall or total objective history evaluated during the study
        :rtype: list(float)
        """
        return self._total_objective_history

    @property
    def best_evaluation_index(self):
        """
        The index of the best evaluation index based on the 
        total objective history. This is the index of the best stored evaluation 
        in order of the study's evaluation history.

        :return: best evaluation index of stored results
        :rtype: int
        """
        return np.argmin(self._total_objective_history)
    
    @property
    def evaluation_ids(self):
        """
        The evaluation id number for each stored result. 
        The id is the evaluation number 
        in order of the study's evaluation history.
        """
        return self._evaluation_ids

    @property
    def best_evaluation_id(self):
        """
        The id of the best evaluation based on the 
        total objective history. The id is the evaluation number 
        in order of the study's evaluation history.

        :return: best evaluation id number
        :rtype: int
        """
        return self._evaluation_ids[self.best_evaluation_index]
    
    @property
    def best_total_objective(self):
        """
        The best total objective value obtained during the study

        :return: The best total objective value
        :rtype: float
        """
        return np.min(self._total_objective_history)
    
    def best_evaluation_set_objective(self, model, obj):
        """
        Returns the best evaluation set objective value and its evaluation index/number.
        An evaluation set objective is the total objective value for a given model
        and objective. It includes a summation of the objective for all states and 
        fields for a give evaluation set. 

        :param model: The model or model name of interest
        :type model: str, :class:`~matcal.core.models.ModelBase`
        :param obj: The objective or objective name of interest
        :type obj: str, :class:`~matcal.core.objective.Objective`
        
        :return: The best evaluation set objective value and its index
        :rtype: tuple(float,int)
        """
        summed_state_objs = self.get_evaluation_set_objectives(model, obj)
        best_index = np.argmin(summed_state_objs)
        best_value = np.min(summed_state_objs)
        return best_value, best_index            

    def get_evaluation_set_objectives(self, model, obj):
        """
        Returns the history of the evaluation set objectives
        for a given model and objective.  An evaluation set objective is the 
        total objective value for a given model
        and objective. It includes a summation of the objective for all states and 
        fields for a give evaluation set. They are returned in order of evaluation.

        :param model: The model or model name of interest
        :type model: str, :class:`~matcal.core.models.ModelBase`
        :param obj: The objective or objective name of interest
        :type obj: str, :class:`~matcal.core.objective.Objective`
        
        :return: The summed state objectives
        :rtype: np.ndarray
        """
        eval_key = self.get_eval_set_name(model, obj)
        obj_evals = self.objective_history[eval_key].objectives
        summed_state_objs = np.zeros(len(obj_evals))
        for idx, eval in enumerate(obj_evals):
            for state in eval:
                for data in eval[state]:
                    for field in data.field_names:
                        summed_state_objs[idx] += data[field]
        return summed_state_objs

    def get_objectives_for_model(self, model):
        """
        Returns the list of objective names that 
        are in the results for a given model. This can be used
        to easily loop over all objectives for the model 
        if processing results for only a given model.

        :param model: The model or model name of interest
        :type model: str, :class:`~matcal.core.models.ModelBase`
        :return: The list of objectives names for the model
        :rtype: list(str)
        """
        model = _get_obj_name_if_not_string(model)
        objs = []
        for eval_set in self.evaluation_sets:
            if model in eval_set:
                model_name, obj_name = self.decompose_evaluation_name(eval_set)
                objs.append(obj_name)
        return objs
    
    @property
    def models_in_results(self):
        """
        Returns the list of unique model names that 
        are in the results. 

        :return: The list of model names included in the results.
        :rtype: set(str)
        """
        models = []
        for eval_set in self.evaluation_sets:
                model_name, obj_name = self.decompose_evaluation_name(eval_set)
                models.append(model_name)
        return set(models)

    def best_simulation_data(self, model, state):
        """
        Returns the best simulation data for a given model and state. 
        It returns a :class:`~matcal.core.data.Data` object with the 
        the best simulation's results. 

        :param model: The model or model name of interest
        :type model: str, :class:`~matcal.core.models.ModelBase`
        :param state: The objective or objective name of interest
        :type state: str, :class:`~matcal.core.state.State`
        
        :return: The best simulation data
        :rtype: :class:`matcal.core.data.Data`
        """
        model_name = _get_obj_name_if_not_string(model)
        best_eval = self.best_evaluation_index
        return self.simulation_history[model_name][state][best_eval]

    def get_experiment_qois(self, model, obj, state, index=None):
        """
        Returns the experiment qois for a given model and state. 

        :param model: The model or model name of interest
        :type model: str, :class:`~matcal.core.models.ModelBase`
        :param state: The objective or objective name of interest
        :type state: str, :class:`~matcal.core.state.State`
        :param index: The index of the desired repeat if given
        :type index: int
        :return: The experiment QoIs
        :rtype: :class:`~matcal.core.data.Data` or list(:class:`~matcal.core.data.Data`)
        """
        model_name = _get_obj_name_if_not_string(model)
        obj_name = _get_obj_name_if_not_string(obj)
        eval_key = self.get_eval_set_name(model_name, obj_name)
        data_list = self._qoi_history[eval_key].experiment_qois[state]
        return _get_return_data_list_history(data_list, index, 
                                             err_msg="study_results.get_experiment_qois")
    
    def get_experiment_data(self, model, obj, state, index=None):
        """
        Returns the experiment data for a given model and state. 
  
        :param model: The model or model name of interest
        :type model: str, :class:`~matcal.core.models.ModelBase`
        :param state: The objective or objective name of interest
        :type state: str, :class:`~matcal.core.state.State`
        :param index: The index of the desired repeat if given
        :type index: int
        :return: The experiment data
        :rtype: :class:`~matcal.core.data.Data` or list(:class:`~matcal.core.data.Data`)
        """
        model_name = _get_obj_name_if_not_string(model)
        obj_name = _get_obj_name_if_not_string(obj)
        eval_key = self.get_eval_set_name(model_name, obj_name)
        data_list = self._qoi_history[eval_key].experiment_data[state]
        return _get_return_data_list_history(data_list, index,
                                             err_msg="study_results.get_experiment_data")
    
    def best_simulation_qois(self, model, obj, state, index=None):
        """
        Returns the best simulation QoIs for a given model and state. 

        :param model: The model or model name of interest
        :type model: str, :class:`~matcal.core.models.ModelBase`
        :param state: The objective or objective name of interest
        :type state: str, :class:`~matcal.core.state.State`
        :param index: The index of the desired repeat if given
        :type index: int
        :return: The best simulation qois
        :rtype: :class:`~matcal.core.data.Data` or list(:class:`~matcal.core.data.Data`)
        """
        best_eval = self.best_evaluation_index
        model_name = _get_obj_name_if_not_string(model)
        obj_name = _get_obj_name_if_not_string(obj)
        
        eval_key = self.get_eval_set_name(model_name, obj_name)
        data_list = self._qoi_history[eval_key].simulation_qois[best_eval][state]
        return _get_return_data_list_history(data_list, index, 
                                             err_msg="study_results.best_simulation_qois")

    def best_residuals(self, model, obj, state, index=None):
        """
        Returns the best objective residual data for a given model, 
        objective and state. 

        :param model: The model or model name of interest
        :type model: str, :class:`~matcal.core.models.ModelBase`
        :param state: The objective or objective name of interest
        :type state: str, :class:`~matcal.core.state.State`
        :param index: The index of the desired repeat if given
        :type index: int
        :return: The best objective residuals
        :rtype: :class:`~matcal.core.data.Data` or list(:class:`~matcal.core.data.Data`)
        """
        best_eval = self.best_evaluation_index
        model_name = _get_obj_name_if_not_string(model)
        obj_name = _get_obj_name_if_not_string(obj)
        eval_key = self.get_eval_set_name(model_name, obj_name)
        data_list = self._obj_history[eval_key].residuals[best_eval][state]
        return _get_return_data_list_history(data_list, index, 
                                             err_msg="study_results.best_residuals")


    def best_weighted_conditioned_residuals(self, model, obj, state, index=None):
        """
        Returns the best objective weighted and conditioned 
        residual data for a given model, 
        objective and state. 

        :param model: The model or model name of interest
        :type model: str, :class:`~matcal.core.models.ModelBase`
        :param state: The objective or objective name of interest
        :type state: str, :class:`~matcal.core.state.State`
        :param index: The index of the desired repeat if given
        :type index: int
        :return: The best objective weighted and conditioned residuals
        :rtype: :class:`~matcal.core.data.Data` or list(:class:`~matcal.core.data.Data`)
        """
        best_eval = self.best_evaluation_index
        model_name = _get_obj_name_if_not_string(model)
        obj_name = _get_obj_name_if_not_string(obj)
        eval_key = self.get_eval_set_name(model_name, obj_name)
        data_list = self._obj_history[eval_key].weighted_conditioned_residuals[best_eval][state]
        err_msg="study_results.best_weighted_conditioned_residuals"
        return _get_return_data_list_history(data_list, index, 
                                             err_msg=err_msg)

    def save(self, filename):
        """
        Saves the study results to a file using 
        :func:`~matcal.core.serializer_wrapper.matcal_save`.
        You should use a "joblib" filename.

        :param filename: The name of the file to save the results to
        :type filename: str
        """
        matcal_save(filename, self)
           
    def _export_parameter_results(filename:str, evaluation_data:dict, evaluation_parameters:dict):
        parameter_string = __class__._make_parameter_string(evaluation_parameters)
        data_keys = list(evaluation_data.keys())
        key_string = __class__._make_key_string(data_keys)
        n_rows = len(evaluation_data[data_keys[0]])
        
        with open(filename, 'w') as eval_file:
            eval_file.write(parameter_string)
            eval_file.write(key_string)
            for row_i in range(n_rows):
                row_string = __class__._make_row_string(evaluation_data, data_keys, row_i)
                eval_file.write(row_string)

    def _make_row_string(evaluation_data, data_keys, row_i):
        row_string = ""
        for key_i, key in enumerate(data_keys):
            if key_i > 0:
                row_string += ", "
            row_string += f"{evaluation_data[key][row_i]}"
        row_string += "\n"
        return row_string

    def _make_key_string(data_keys):
        key_string = ""
        for key_idx, key in enumerate(data_keys):
            if key_idx > 0:
                key_string += ", "
            key_string += key
        key_string += "\n"
        return key_string
    
    def _make_parameter_string(evaluation_parameters):
        parameter_string = "{"
        for param_idx, (name, value) in enumerate(evaluation_parameters.items()):
            if param_idx > 0:
                parameter_string += ", "
            parameter_string += f"\"{name}\":{value}"
        parameter_string += "}\n"
        return parameter_string       
           
    def _set_outcome(self, outcome_dict:dict):
        if not isinstance(outcome_dict, (dict, OrderedDict)):
            err_msg = f"Outcome must be of type dict or OrderedDict, passed type: {type(outcome_dict)}"
            raise TypeError(err_msg)
        self._outcome = OrderedDict()
        for name, value in outcome_dict.items():
            self._outcome[name] = value
        self._convert_outcome_to_attributes()
    
    class ResultsAttribute():
        """
        Study specific results are stored as attributes are 
        a special class that have methods aiding in results 
        organization.
        """
        def _get_reportable_attributes(self):
            all_attributes = self.__dict__.keys()
            reportable_attributes = [attr for attr in all_attributes if not attr.startswith('__') and not callable(getattr(self, attr))]
            return reportable_attributes

        def __str__(self):
            attributes = self._get_reportable_attributes()
            attributes_str = "\n".join(f"{attr}: {getattr(self, attr)}" for attr in attributes)
            return attributes_str

        def to_dict(self):
            """
            Converts a result attribute into a dictionary 
            with the result name as the key and the result value 
            as the value.

            :rtype: OrderedDict
            """
            attributes = self._get_reportable_attributes()
            attributes_dict = OrderedDict({attr: getattr(self, attr) for attr in attributes})
            
            return attributes_dict
        
        def to_list(self, attributes=None):
            """
            Returns a result attribute as a list. Can be useful 
            when needing to pass results to other methods of functions 
            that require unnamed values.

            :rtype: list
            """
            attributes = self._get_reportable_attributes()
            attributes_list = [getattr(self, attr) for attr in attributes]
            return attributes_list

    def _convert_outcome_to_attributes(self):
        for outcome_key, outcome_value in self._outcome.items():
            split_key = outcome_key.split(":")
            key_depth = len(split_key)
            previous_base = self
            for depth in range(key_depth):
                base_key = split_key[depth]
                base_value = self._determine_value(outcome_value, key_depth, depth)
                if not hasattr(previous_base, base_key) or depth == key_depth-1:
                    setattr(previous_base, base_key, base_value)                    
                previous_base = getattr(previous_base, base_key)

    def _determine_value(self, outcome_value, key_depth, depth):
        if depth == key_depth - 1:
            base_value = outcome_value
        else:
            base_value = self.ResultsAttribute()
        return base_value
        
    def _initialize_exit_status(self, success, exit_message):
        if success:
            self._exit_message = "Successful:\n"
        else:
            self._exit_message = "Failed   :\n"
        self._exit_message += exit_message
        
    def _update_parameter_history(self, batch_parameters, eval_order):
        if len(self._parameter_history) < 1:
            self._initialize_parameter_history(batch_parameters, eval_order)
        for idx, key in enumerate(eval_order):
            if idx % self._save_freq == 0:
                for p_name, p_val in batch_parameters[key].items(): 
                    self._parameter_history[p_name].append(p_val)

    def _update_simulation_history(self, simulation_results_dc, model_name):
        if self._record_data:
            if model_name not in self._simulation_history:
                self._simulation_history[model_name] = DataCollection(f"{model_name} simulation history")
            _update_data_collection(self._simulation_history[model_name], 
                                            simulation_results_dc)

    def _get_eval_id_number(self, eval):
        eval_id_number = int(eval.split(".")[-1])
        self._number_of_evaluations = max(eval_id_number, 
                                          self._number_of_evaluations)
        return eval_id_number

    def _update_results_history(self, raw_obj, total_obj, all_qois, eval_order):
        for idx, eval in enumerate(eval_order):
            eval_id_number = self._get_eval_id_number(eval)
            if idx % self._save_freq == 0:
                self._total_objective_history.append(total_obj[eval]) 
                self._evaluation_ids.append(eval_id_number)
                for model_name in all_qois[eval].keys():
                    sim_data = self._get_simulation_results_from_qois(all_qois[eval][model_name])
                    self._update_simulation_history(sim_data, model_name)
                    for objective_name in all_qois[eval][model_name].keys():
                        eval_set_name = self.get_eval_set_name(model_name, objective_name)
                        self._update_qoi_history(eval_set_name, 
                                                 all_qois[eval][model_name][objective_name])
                        self._update_objective_history(eval_set_name, 
                                                       raw_obj[eval][model_name][objective_name])

    def _get_simulation_results_from_qois(self, model_obj_set_qois):
        first_obj_key = list(model_obj_set_qois.keys())[0]
        simulation_results_dc = model_obj_set_qois[first_obj_key].simulation_data
        return simulation_results_dc

    def _update_qoi_history(self, eval_set_name, obj_qois):
        self._qoi_history[eval_set_name]._set_exp_information(obj_qois.experiment_data,
                                                              obj_qois.experiment_qois,
                                                              obj_qois.weighted_conditioned_experiment_qois)
        self._qoi_history[eval_set_name]._update_sim_information(obj_qois.simulation_qois, 
                                                                 obj_qois.weighted_conditioned_simulation_qois)
    def _update_objective_history(self, eval_set_name, obj_results):
        self._obj_history[eval_set_name]._update_residuals(obj_results.residuals, 
                                                           obj_results.weighted_conditioned_residuals)
        self._obj_history[eval_set_name]._update_objectives(obj_results.objectives)

    def _initialize_parameter_history(self, batch_parameters, key_order):
        first_key = key_order[0]
        for param_name in batch_parameters[first_key].keys():
            self._parameter_history[param_name] = []
            
    def _initialize_evaluation_sets(self, evaluation_sets_qois, key_order):
        if len(self._evaluation_sets) < 1:
            first_eval = key_order[0]
            model_objectives = evaluation_sets_qois[first_eval]
            for model_name, objective_set in model_objectives.items():
                for objective_name, qoi_results in objective_set.items():
                    eval_set_name = self.get_eval_set_name(model_name, objective_name)
                    if eval_set_name not in self._evaluation_sets:
                        self._evaluation_sets.append(eval_set_name)
            self._initialize_objective_information()
            self._initialize_qoi_information()

    def get_eval_set_name(self, model, objective):
        """
        Returns the evaluation set name for a given model and 
        objective which is needed to access certain results. 

        :param model: The model or model name of interest
        :type model: str, :class:`~matcal.core.models.ModelBase`
        :param obj: The objective or objective name of interest
        :type obj: str, :class:`~matcal.core.objective.Objective`
        
        :return: The evaluation set name
        :rtype: str
        """
        model = _get_obj_name_if_not_string(model)
        objective = _get_obj_name_if_not_string(objective)
        eval_set_name = f"{model}:{objective}"
        return eval_set_name
    
    def decompose_evaluation_name(self, evaluation_set_name):
        """
        Returns the model name and objective name 
        given an evaluation set name.

        :param evaluation_set_name: The model or model name of interest
        :type evaluation_set_name: str
        
        :return: The model name and objective name
        :rtype: tuple(str,str)
        """
        split_name = evaluation_set_name.split(':')
        return split_name
    
    def _initialize_objective_information(self):
        self._obj_history = OrderedDict()
        for eval_set_name in self._evaluation_sets:
            self._obj_history[eval_set_name] = ObjectiveInformation(record_objectives=self._record_objectives, 
                                                                    record_residuals=self._record_residuals,
                                                                    record_weighted_conditioned=
                                                                    self._record_weighted_conditioned)
                    
    def _initialize_qoi_information(self):
        self._qoi_history = OrderedDict()
        for eval_set_name in self._evaluation_sets:
            self._qoi_history[eval_set_name] = QoiInformation(record_data=self._record_data, 
                                                              record_qois=self._record_qois, 
                                                              record_weighted_conditioned=
                                                              self._record_weighted_conditioned)
    

def _get_obj_name_if_not_string(obj):
    if not isinstance(obj, str):
        return obj.name
    else:
        return obj


def _get_return_data_list_history(data_list, repeat_index=None,
                                  err_msg="study_results.get_*_qois/data"):
    return_val = data_list
    if repeat_index is not None:
        check_value_is_positive_integer(repeat_index, "index", err_msg)
        if repeat_index >= len(return_val):
            index_error_msg = (f"The index {repeat_index} is too large for the results"
                           f" list being returned from {err_msg}")
            raise IndexError(index_error_msg)
        return_val = return_val[repeat_index]
    return return_val


def _update_data_collection(dc_to_update, source_dc):
    for state in source_dc:
        for data in source_dc[state]:
            dc_to_update.add(data)

    
class ObjectiveInformation:
    """
    Contains the objective information for a study. 
    Objective information includes the objectives, 
    the residuals and the weighted and conditioned residuals 
    as list of :class:`~matcal.core.data.DataCollection` objects.

    :ivar residuals: A list of residuals in order of parameter evaluation for
        the study.
    :vartype residuals: list(:class:`~matcal.core.data.DataCollection`)
    :ivar weighted_conditioned_residuals: list(:class:`~matcal.core.data.DataCollection`)
        A list of weighted and conditioned residuals in order of parameter evaluation for
        the study.
    :vartype weighted_conditioned_residuals: list(:class:`~matcal.core.data.DataCollection`)

    :ivar objectives: list(:class:`~matcal.core.data.DataCollection`)
        A list of objectives in order of parameter evaluation for
        the study.
    :vartype objectives: list(:class:`~matcal.core.data.DataCollection`)
    """
    def __init__(self, record_objectives=True, record_residuals=True, 
                record_weighted_conditioned=False):
        self.residuals = []
        self.weighted_conditioned_residuals = []
        self.objectives = [] 
        self._record_objectives = record_objectives
        self._record_residuals = record_residuals
        self._record_weighted_conditioned = record_weighted_conditioned
        
    def _update_residuals(self, new_residuals, new_wc_residuals):
        if self._record_residuals:
            self.residuals.append(new_residuals)
        if self._record_residuals and self._record_weighted_conditioned:
            self.weighted_conditioned_residuals.append(new_wc_residuals)
            
    def _update_objectives(self, objectives):
        if self._record_objectives:
            self.objectives.append(objectives)
   
 
class QoiInformation:
    """
    Contains the QoI information for a study. 
    QoI information includes the experiment data, 
    the experiment QoIs, and the weighted and conditioned experiment QoIs
    as :class:`~matcal.core.data.DataCollection` objects.

    It also contains the simulation QoIs and simulation weighted and conditioned 
    QoIs as lists of :class:`~matcal.core.data.DataCollection` objects.

    :ivar experiment_data: The experimental data
    :vartype experiment_data: :class:`~matcal.core.data.DataCollection`
    :ivar experiment_qois: The experimental QoIs
    :vartype experiment_qois: :class:`~matcal.core.data.DataCollection`
    :ivar experiment_weighted_conditioned_qois: The weighted and conditioned
        experiment QoIs
    :vartype experiment_weighted_conditioned_qois: :class:`~matcal.core.data.DataCollection`
    
    :ivar simulation_qois: The simulation QoIs in order of parameter evaluation for
        the study.
    :vartype simulation_qois: list(:class:`~matcal.core.data.DataCollection`)
    :ivar simulation_weighted_conditioned_qois: list(:class:`~matcal.core.data.DataCollection`)
        The weighted and conditioned simulation QoIs in order of parameter evaluation for
        the study.
    :vartype simulation_weighted_conditioned_qois: list(:class:`~matcal.core.data.DataCollection`)
    """
    
    def __init__(self, record_data=True, record_qois=True, 
        record_weighted_conditioned=False):
        self.experiment_data = DataCollection("experimental data")
        self.experiment_qois = DataCollection("experimental qois")
        self.experiment_weighted_conditioned_qois = DataCollection("experiment weighted "
                                                                   "conditioned qois")
        self._experiment_data_set = False
        self.simulation_qois = []
        self.simulation_weighted_conditioned_qois =[]
        self._record_data = record_data
        self._record_qois = record_qois
        self._record_weighted_conditioned = record_weighted_conditioned

    def _set_exp_information(self, exp_data, exp_qois, exp_wc_qois):
        if not self._experiment_data_set:
            if self._record_data:
                _update_data_collection(self.experiment_data, 
                                                   exp_data)
            if self._record_qois:
                _update_data_collection(self.experiment_qois, 
                                                   exp_qois)
            if self._record_weighted_conditioned and self._record_qois:
                _update_data_collection(self.experiment_weighted_conditioned_qois, 
                                        exp_wc_qois)
            self._experiment_data_set=True

    def _update_sim_information(self, sim_qois, sim_wc_qois):
        if self._record_qois:
            self.simulation_qois.append(sim_qois)
        if self._record_qois and self._record_weighted_conditioned:
            self.simulation_weighted_conditioned_qois.append(sim_wc_qois)
        
        
def _record_results(results:StudyResults, formatted_parameters, raw_obj, total_obj, qois, skip):
    if not skip:
        ordered_evaluation_keys = _sort_workdirs(list(formatted_parameters.keys()))
        results._initialize_evaluation_sets(qois, ordered_evaluation_keys)
        results._update_parameter_history(formatted_parameters, ordered_evaluation_keys)
        results._update_results_history(raw_obj, total_obj, qois, ordered_evaluation_keys)
        results.save(IN_PROGRESS_RESULTS_FILENAME+".joblib")   
        

def _unpack_evaluation(batch_results):
    batch_raw_objectives = batch_results[0]
    total_objectives = batch_results[1]
    batch_qois = batch_results[2]
    return batch_raw_objectives,total_objectives,batch_qois