from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import os
import time

from matcal.core.computing_platforms import MatCalJobDispatchDelayFunctionIdentifier
from matcal.core.data import DataCollection
from matcal.core.logger import initialize_matcal_logger
from matcal.core.simulators import SimulatorFailureResults, SimulatorResults


logger = initialize_matcal_logger(__name__)


class Job():
    def __init__(self, compute_task, cores, parameters, working_dir=None, submit_delay=None):
        self._compute_task = compute_task
        self._cores = cores
        self._submit_delay = submit_delay
        self._parameters = parameters
        self._working_dir = working_dir

    @property
    def task(self):
        return self._compute_task

    def run(self):
        init_dir = os.getcwd()
        working_dir = None
        if self._working_dir is not None and os.path.exists(self._working_dir):
            working_dir = self._working_dir
        results = self._compute_task.run(parameters=self._parameters, working_dir=working_dir)
        return results

    @property
    def cores(self):
        return self._cores

    @property
    def submit_delay(self):
        if self._submit_delay is None:
            return 0
        else:
            return self._submit_delay

def prepare_parameter_evaluation_jobs(evaluation_sets, parameters, evaluation_name):
    bill_of_jobs = []
    for model, evaluation_set in evaluation_sets.items():
        delay_time = _get_delay_time()
        for state, current_simulation in evaluation_set.simulators.items():

            current_job = Job(current_simulation, model.number_of_local_cores, 
                              parameters, working_dir=evaluation_name, submit_delay=delay_time)
            key = _make_key(evaluation_name, model.name, state.name)
            bill_of_jobs.append([key, current_job])
    return bill_of_jobs

def _get_delay_time():
    delay_time_func =  MatCalJobDispatchDelayFunctionIdentifier.identify()
    delay_time = delay_time_func()
    return delay_time

def _make_key(work_dir, model_name, state_name):
    return (work_dir, model_name, state_name)

def _string_key(key):
    delim = "    "
    s = delim
    names = ["Evaluation", "Model", "State"]
    for item, name in zip(key, names):
        s += f"{name}: {item}" + delim
    return s

def _break_key(key):
    work_dir = key[0]
    model = key[1]
    state = key[2]
    return work_dir, model, state

def run_jobs_serial(bill_of_jobs, batch_restart):
    raw_results = []
    for job_key, job in bill_of_jobs:
        previous_results_filename = batch_restart.retrieve_results_file(job_key)
        if previous_results_filename is None:
            run_results = job.run()
            if run_results.source_filename is not None:
                batch_restart.record(job_key, run_results.source_filename)
        else:
            run_results = _retrieve_restart_results(job_key, previous_results_filename, job.task)
        raw_results.append((job_key, run_results))
    results = _convert_tuple_to_dict(raw_results)
    return results


def _retrieve_restart_results(job_key, previous_results_filename, simulator):
    logger.info(f"  Restart Results Found:"+ _string_key(job_key))
    try:
        model_results = simulator._results_information.read(previous_results_filename)
    except Exception as e:
        workdir, model, state = job_key
        raise RuntimeError("Could not read existing results from file "+
                          f"\'{previous_results_filename}\' for model \'{model}\' "+
                          f"state \'{state}\' in \'{workdir}\'. "+
                          "Ensure files were not moved "+
                          f"or deleted and check input. "
                          f"Caught the following error:\n {repr(e)}")
    model_results.set_state(simulator.state)
    run_results = SimulatorResults(model_results, '', '', 0)
    return run_results


def dispatch_jobs(bill_of_jobs, max_cores, batch_restart, use_threads=False, 
                  always_use_threads=False):
    _check_cores_in_jobs(bill_of_jobs, max_cores)
    if use_threads:
        use_threads = _use_threads(bill_of_jobs, max_cores, always_use_threads=always_use_threads)

    dispatcher = _create_job_dispatcher(max_cores, use_threads, batch_restart)
    for job_key, job in bill_of_jobs:
        dispatcher.dispatch_job_when_available(job_key, job)
    raw_results = dispatcher.get_results_when_finished()
    results = _convert_tuple_to_dict(raw_results)
    return results


def _use_threads(bill_of_jobs, max_cores_available, always_use_threads=False):
    if always_use_threads:
        return True
    use_threads = True
    for index, jobs in enumerate(bill_of_jobs):
        job = bill_of_jobs[index][1]
        if index < len(bill_of_jobs)-1:
            if job.cores + bill_of_jobs[index+1][1].cores > max_cores_available and use_threads:
                use_threads = False
    return use_threads


def _check_cores_in_jobs(jobs, max_cores):
    for job_key, job in jobs:
        if job.cores > max_cores:
            message = f"The job ({job}) requires {job.cores} cores. "
            message += "This is more cores than the {max_cores} available cores."
            raise TooManyJobCoresError(message)


def _create_job_dispatcher(max_cores, use_threads, batch_restart):
    if use_threads:
        return Dispatcher(max_cores, ThreadPoolExecutor, batch_restart)
    else: 
        return Dispatcher(max_cores, ProcessPoolExecutor, batch_restart)


def _convert_tuple_to_dict(raw_results):
    processed_results = OrderedDict()
    for key, values in raw_results:
        processed_results[key] = values
    return processed_results


def convert_results_to_dict_of_data_collections(raw_results_dict, failure_defaults):
    processed_dict = OrderedDict()
    for key, value in raw_results_dict.items():
        eval_name, model, state = _break_key(key)
        if eval_name not in processed_dict.keys():
            processed_dict[eval_name] = OrderedDict()
        eval_results_dict = processed_dict[eval_name]
        if model not in eval_results_dict:
            eval_results_dict[model] = DataCollection("simulation")
        if isinstance(value, SimulatorFailureResults):
            eval_results_dict[model].add(failure_defaults.make_failure_data(model, value.state))
        else:   
            eval_results_dict[model].add(value.results_data)
    return processed_dict
        

class Dispatcher:

    def __init__(self, max_cores, pool_type, batch_restart):
        self.max_cores = max_cores
        self.current_core_use = 0
        self.pool = pool_type(max_cores)
        self._running_jobs = []
        self._job_results = []
        self._finished = False
        self._batch_restart = batch_restart
    
    def dispatch_job_when_available(self, job_key, job):
        self._confirm_not_finished()
        job_dispatched = False
        while not job_dispatched:
            results_filename = self._batch_restart.retrieve_results_file(job_key)
            if results_filename != None:
                restart_results = _retrieve_restart_results(job_key, results_filename, job.task)
                self._job_results.append([job_key, restart_results])
                job_dispatched = True
            elif self._has_sufficient_cores(job.cores):
                dispatched_job = self._dispatch_job(job)
                self._running_jobs.append([job_key, dispatched_job, job.cores])
                job_dispatched = True
                logger.info("  Dispatched job: "+ _string_key(job_key))
            else:
                logger.info("  Waiting to dispatch job: "+ _string_key(job_key))
                self._wait()

    def get_results_when_finished(self):
        logger.info("  Waiting for jobs to complete")
        while self._has_running_jobs():
            self._wait()
        self._finish()
        return self._job_results

    def _has_sufficient_cores(self, new_cores):
        return self.current_core_use + new_cores <= self.max_cores

    def _dispatch_job(self, job):
        dispatched_job = self._dispatch(job)
        self.current_core_use += job.cores
        time.sleep(job.submit_delay)
        return dispatched_job

    def _dispatch(self, job):
        return self.pool.submit(job.run)

    def _wait(self):
        job_finished = False
        while not job_finished:
            finished_id = self._find_first_finished_job_index()
            if finished_id is not None:
                self._update_with_finished_job(finished_id)
                job_finished = True

    def _find_first_finished_job_index(self):
        for job_index, (job_key, dispathced_job, job_cores) in enumerate(self._running_jobs):
            if dispathced_job.done():
                return job_index
        return None
    
    def _update_with_finished_job(self, job_index):
        job_key, dispatched_job, job_cores = self._running_jobs.pop(job_index)
        self.current_core_use -= job_cores
        logger.debug("  Recording Results: "+ _string_key(job_key))
        job_result = None
        try:
            job_result = dispatched_job.result()
        except Exception as e:
            self._batch_restart.close()
            raise e

        logger.debug(job_result.stdout)
        logger.info(job_result.stderr)
        self._job_results.append([job_key, job_result])
        self._batch_restart.record(job_key, job_result.source_filename)
        logger.info("  Finished job: "+ _string_key(job_key))

    def _has_running_jobs(self):
        remaining_jobs = len(self._running_jobs)
        return  remaining_jobs > 0

    def _finish(self):
        self._finished = True

    class ClosedDispatchError(RuntimeError):

        def __init__(self):
            message = "Attempting to use a dispatcher that is finished"
            super().__init__(message)

    def _confirm_not_finished(self):
        if self._finished:
            raise self.ClosedDispatchError()
    

class TooManyJobCoresError(RuntimeError):
    pass
 
