from abc import ABC, abstractmethod
from collections import OrderedDict
from matcal.core.data import convert_dictionary_to_data
import numpy as np
import os
import shutil

from matcal.core.constants import (BATCH_RESTART_FILENAME, DESIGN_PARAMETER_FILE,  
                                   MATCAL_TEMPLATE_DIRECTORY)
from matcal.core.logger import initialize_matcal_logger
from matcal.core.multi_core_job_pool import (convert_results_to_dict_of_data_collections, 
     dispatch_jobs, prepare_parameter_evaluation_jobs, run_jobs_serial)
from matcal.core.reporter import MatCalParameterReporterIdentifier
from matcal.core.serializer_wrapper import matcal_save
from matcal.core.utilities import remove_directory

logger = initialize_matcal_logger(__name__)

def append_unique_to_list(values, base_list):
    for val in values:
        if val not in base_list:
            base_list.append(val)
    return base_list


class EvaluationFailureDefaults():
    
    def __init__(self, evauluation_sets):
        self.all_fallback_fieldnames = []
        self.model_specific_fallbacks = {}
        self._initialze_fallback_fields_from_eval_sets(evauluation_sets)
        self._initialize_fallback_specific_model_values(evauluation_sets)

    def make_failure_data(self, model_name, state):
        data = self._populate_model_specific_data(model_name)
        n_points = self._identify_number_of_points(data)
        self._populate_generic_data(data, n_points)
        fake_data = convert_dictionary_to_data(data)
        fake_data.set_state(state)
        return fake_data

    def _populate_generic_data(self, data, n_points):
        generic_values = np.linspace(-1, 1, n_points)
        for fname in self.all_fallback_fieldnames:
            if fname not in data.keys():
                missing_msg = f"Evaluation Failed and no failure defaults detected for field {fname}, resorting to generic defaults."
                logger.warning(missing_msg)
                data[fname] = generic_values

    def _populate_model_specific_data(self, model_name):
        data = {}
        model_defaults = self.model_specific_fallbacks[model_name]
        for fname, fvalue in model_defaults.items():
            data[fname] = fvalue       
        return data

    def _identify_number_of_points(self, data_dict):
        if len(data_dict) > 0:
            first_value_length = len(next(iter(data_dict.values())))
            return first_value_length
        else:
            default_value = 20
            return default_value

    def _initialze_fallback_fields_from_eval_sets(self, evaluation_sets):
        for eval_set in evaluation_sets.values():
            for objective_set in eval_set.objective_sets:
                for objective in objective_set.objectives.values():
                    append_unique_to_list(objective._required_fields, self.all_fallback_fieldnames)

    def _initialize_fallback_specific_model_values(self, evaluation_sets):
        for model in evaluation_sets:
            self.model_specific_fallbacks[model.name] = model.on_failure_values()


class BatchRestartBase(ABC):

    @abstractmethod
    def record(self, job_keys:list, results_filename:str)->None:
        """"""
    
    @abstractmethod
    def _retrieve_results_file_impl(self, job_keys:list)->str:
        """"""

    @abstractmethod
    def close(self)->None:
        """"""

    @abstractmethod
    def file_extension(self)->str:
        """"""

    def __init__(self, save_only:bool):
        self._save_only = save_only

    def _create_h5_group(self, job_keys:list)->str:
        group_name = ""
        for i, key_element in enumerate(job_keys):
            if i > 0:
                group_name += "/"
            group_name += f"{key_element}"
        return group_name
    
    @property
    def default_lookup_return(self):
        return None
        
    def retrieve_results_file(self, job_keys:list)->str:
        if self._save_only:
            return self.default_lookup_return
        return self._retrieve_results_file_impl(job_keys)


class BatchRestartNone(BatchRestartBase):
    # Used to turn off file saving which may interfere with 3rd party libraries. 

    def record(self, job_keys, results_filename):
        print("Batch Restart Recording")

    def _retrieve_results_file_impl(self, job_keys):
        print("Batch Restart Retrieving")
        return self.default_lookup_return

    def file_extension():
        return None

    def close(self):
        print("Batch Restart Closing")


class BatchRestartHDF5(BatchRestartBase):

    def __init__(self, save_only:bool)->None:
        import h5py
        super().__init__(save_only)
        if save_only:
            write_or_append = "w"
        else:
            write_or_append = "r+" #r+ is read/write/append, will not make a new file
        self._restart_file = h5py.File(BATCH_RESTART_FILENAME+self.file_extension(), write_or_append)

    def record(self, job_keys:list, results_filename:str)->None:
        if not isinstance(results_filename, str):
            return None
        group_name = self._create_h5_group(job_keys)
        self._restart_file.create_group(group_name)
        self._restart_file[group_name].create_dataset('results', data=[results_filename])

    def _retrieve_results_file_impl(self, job_keys:list)->str:
        group_name = self._create_h5_group(job_keys)
        if group_name in self._restart_file:
            res_filename = self._restart_file[group_name]['results'][0].decode('ascii')
        else:
            res_filename = self.default_lookup_return
        return res_filename

    def close(self):
        self._restart_file.close()

    @staticmethod
    def file_extension():
        return ".h5"


class BatchRestartCSV(BatchRestartBase):

    def __init__(self, save_only):
        super().__init__(save_only)
        current_dir = os.getcwd()
        full_path_batch_restart_filename = os.path.join(current_dir, BATCH_RESTART_FILENAME+self.file_extension())
        self._finished_jobs = {}
        if save_only:
            write_or_append = "w"
        else:
            write_or_append = "a" 
            self._finished_jobs = self._get_finished_jobs_info(full_path_batch_restart_filename)
        self._restart_file = open(full_path_batch_restart_filename, write_or_append)

    def _get_finished_jobs_info(self, full_path_batch_restart_filename):
        finished_jobs = {}
        with open(full_path_batch_restart_filename, "r") as f:
            for line in f.readlines():
                job_key, results_filename = line.split(",")
                finished_jobs[job_key] = results_filename.strip()
        return finished_jobs
    
    def record(self, job_keys:list, results_filename:str)->None:
        if not isinstance(results_filename, str):
            return None
        group_name = self._create_h5_group(job_keys)
        self._finished_jobs[group_name] = results_filename
        self._restart_file.write(f'{group_name},{results_filename}\n') 
        self._restart_file.flush()

    def _retrieve_results_file_impl(self, job_keys:list)->str:
        group_name = self._create_h5_group(job_keys)
        if group_name in self._finished_jobs:
            res_filename = self._finished_jobs[group_name]
        else:
            res_filename = self.default_lookup_return
        return res_filename

    def close(self):
        self._restart_file.close()

    @staticmethod
    def file_extension():
        return ".csv"

SelectedBatchRestartClass = BatchRestartCSV


class ParameterBatchEvaluator():
    """
    Class in charge of running the requested parameter evaluations for a parameter study. 
    """

    def __init__(self, total_cores, study_evaluation_sets,  
                 use_threads=False, always_use_threads=False, run_async=True):
        self._total_study_cores = None
        self._init_dir = os.getcwd()
        self._total_study_cores = total_cores
        self._evaluation_sets = study_evaluation_sets
        self._use_threads = use_threads
        self._always_use_threads = always_use_threads
        self._run_async = run_async                
        self._failure_defaults = EvaluationFailureDefaults(study_evaluation_sets)
        
    def _make_objective_results_dict(self, sim_runner_results):
        study_objective_results = OrderedDict()
        study_objective_qois = OrderedDict()
        for eval_set in self._evaluation_sets.values():
            eval_set_sim_results = sim_runner_results[eval_set.model.name]
            eval_set_objective_results, eval_set_objective_qois = eval_set.evaluate_objectives(eval_set_sim_results)
            study_objective_results[eval_set.model.name] = eval_set_objective_results
            study_objective_qois[eval_set.model.name] = eval_set_objective_qois
        return study_objective_results, study_objective_qois

    def run(self, parameter_sets, is_residual_study):
        objectives, total_objectives, qois = self.evaluate_parameter_batch(parameter_sets, is_residual_study, False)
        return self.default_results_formatter(objectives, total_objectives, 
                                              parameter_sets, qois)

    def restart_run(self, parameter_sets, is_residual_study):
        objectives, total_objectives, qois = self.evaluate_parameter_batch(parameter_sets, is_residual_study, True)
        return self.default_results_formatter(objectives, total_objectives, 
                                              parameter_sets, qois)
    
    @staticmethod
    def default_results_formatter(batch_raw_objectives, total_objs, 
                                  parameter_sets, qois):
        results = {'parameters':parameter_sets, 
                   'objectives':list(batch_raw_objectives.values()), 
                   'total_objectives':list(total_objs.values()),
                   'qois':list(qois.values())}
        return results

    def evaluate_parameter_batch(self, parameter_sets, is_residual_study, is_restart=False):
        save_only = not is_restart
        batch_restart = SelectedBatchRestartClass(save_only)
        objectives, total_objectives, qois = self._evaluate_parameter_batch_impl(parameter_sets, 
                                                                     is_residual_study, batch_restart, is_restart)
        batch_restart.close()

        return objectives, total_objectives, qois

    def _evaluate_parameter_batch_impl(self, parameter_sets, is_residual_study, batch_restart, is_restart):
        bill_of_jobs = self._assemble_jobs(parameter_sets, is_restart)
        job_results = self._run_bill_of_jobs(bill_of_jobs, batch_restart)
        objectives, qois, total_objectives = self._process_job_results(parameter_sets, is_residual_study, job_results)
        
        return objectives, total_objectives, qois

    def _run_bill_of_jobs(self, jobs_to_run, batch_restart):
        job_results = self._run_jobs(jobs_to_run, batch_restart)
        logger.info(f"\nAll parameter set evaluations for the current  batch completed.\n")        
        job_results = convert_results_to_dict_of_data_collections(job_results, self._failure_defaults)
        return job_results

    def _process_job_results(self, parameter_sets, is_residual_study, job_results):
        objectives, qois = self._calculate_objectives_from_batch_results(job_results, parameter_sets)
        total_objectives = _calculate_total_objective(objectives, is_residual_study)
        return objectives,qois,total_objectives

    def _assemble_jobs(self, parameter_sets, is_restart):
        num_param_sets_to_evaluate = len(parameter_sets)
        logger.info(f"Preparing to evaluate {num_param_sets_to_evaluate} parameter sets in the current batch...")
        jobs_to_run = self._prepare_jobs_to_run_for_parameter_batch(parameter_sets, is_restart)
        return jobs_to_run

    def _run_jobs(self, jobs_to_run, batch_restart):
        if self._run_async:
            job_results = dispatch_jobs(jobs_to_run, self._total_study_cores, batch_restart,
                                        self._use_threads, self._always_use_threads)
        else:
            job_results = run_jobs_serial(jobs_to_run, batch_restart)
        return job_results
    
    def _calculate_objectives_from_batch_results(self, job_results, parameter_sets):
        objectives = OrderedDict()
        qois = OrderedDict()
        for evaluation_name in parameter_sets.keys():
            logger.debug(f"extracting objectives for workdir: {evaluation_name}")
            objectives[evaluation_name], qois[evaluation_name] = self._make_objective_results_dict(job_results[evaluation_name])
            _log_evaluation_set_results(evaluation_name, objectives[evaluation_name])
        return objectives, qois
    
    def _prepare_jobs_to_run_for_parameter_batch(self, parameter_sets, is_restart):
        bill_of_jobs = []
        for evaluation_name, parameter_set in parameter_sets.items():
            _setup_workdir(evaluation_name, is_restart)
            write_parameter_include_file(parameter_set, evaluation_name)

            workdir_bill_of_jobs = prepare_parameter_evaluation_jobs(self._evaluation_sets,
                                                        parameter_set, evaluation_name)
            bill_of_jobs += workdir_bill_of_jobs
        return bill_of_jobs
        
def _setup_workdir(workdir_name, is_restart):
    if os.path.exists(workdir_name):
        if is_restart:
            return None
        else:
            remove_directory(workdir_name)
    if os.path.exists(MATCAL_TEMPLATE_DIRECTORY):
        shutil.copytree(MATCAL_TEMPLATE_DIRECTORY, workdir_name, 
                        symlinks=True)


def write_parameter_include_file(params, path):
    if os.path.exists(path):
        dictionary_reporter = MatCalParameterReporterIdentifier.identify()
        dictionary_reporter(os.path.join(path, DESIGN_PARAMETER_FILE), params)


def _log_evaluation_set_results(evaluation_dir, eval_set_results):
    logger.info(f"\tEvaluation results for \"{evaluation_dir}\":")
    for model in eval_set_results.keys():
        for objective in eval_set_results[model].keys():
            logger.info(f"\t\tObjective \"{objective}\" for model \"{model}\" = {eval_set_results[model][objective].get_objective()}")
    logger.info("")


def _calculate_total_objective(objectives_dict, is_residual_study):
    combined_objs, combined_resds, evaluation_names = flatten_evaluation_batch_results(objectives_dict)
    if is_residual_study:
        total_objectives = _get_total_objective_from_residuals_from_each_evaluation(combined_resds, evaluation_names)
    else:
        total_objectives = _get_total_objective_from_objectives_from_each_evaluation(combined_objs, evaluation_names)
    return total_objectives


def _get_total_objective_from_residuals_from_each_evaluation(residuals_for_all_evals, evaluation_names):
    total_objectives = {}
    for residual_list, eval_name in zip(residuals_for_all_evals, evaluation_names):
        total_objectives[eval_name] = float(np.linalg.norm(residual_list)**2)
    return total_objectives
    

def _get_total_objective_from_objectives_from_each_evaluation(objectives_for_all_evals, evaluation_names):
    total_objectives = {}
    for residual_list, eval_name in zip(objectives_for_all_evals, evaluation_names):
        total_objectives[eval_name] = float(np.sum(residual_list))
    return total_objectives


def flatten_evaluation_batch_results(new_batch_results):
    combined_objs = []
    combined_resds = []
    evaluation_dir_names = []
    for evaluation_dir, evaluation_set_results in new_batch_results.items():
        evaluation_dir_names.append(evaluation_dir)
        combined_objs.append(_combine_objective_results(evaluation_set_results))
        combined_resds.append(_combine_residual_results(evaluation_set_results))
   
    return combined_objs, combined_resds, evaluation_dir_names


def _combine_objective_results(results_dict):
    combined_objectives = []

    for model in results_dict.keys(): 
        for result in results_dict[model].values():
            combined_objectives.append(result.get_objective())
    return np.array(combined_objectives)


def _combine_residual_results(results_dict):
    combined_residuals = np.array([])
    for model in results_dict.keys(): 
        for result in results_dict[model].values():
            combined_residuals = np.append(combined_residuals,
                                           np.asarray(result.calibration_residuals))
    return combined_residuals


class MissingKeyError(RuntimeError):

    def __init__(self, missing_key):
        message = f"ERROR :: Missing Key: {missing_key}"
        super().__init__(message)
