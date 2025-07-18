from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
import os
from queue import Queue, Empty
import subprocess
from time import sleep

from matcal.core.computing_platforms import (HPCComputingPlatform, 
    LocalComputingPlatform, RemoteComputingPlatform, local_computer)
from matcal.core.logger import initialize_matcal_logger
from matcal.core.object_factory import (IdentifierByTestFunction, ObjectCreator, 
                                       SpecificObjectFactory)


logger = initialize_matcal_logger(__name__)


class ExternalExecutableBase(ABC):

    class ExternalExecutableError(RuntimeError):
        def __init__(self, *args):
            super().__init__(*args)

    def __init__(self, commands, environment_commands=None, 
                 computer=local_computer, 
                 working_directory=None):
        self._commands = commands
        self._environment_commands = environment_commands
        if self._environment_commands is None:
            self._environment_commands = []
        self._computer = computer
        self._live_logger = False
        self._process = None
        self._q_stdout = None
        self._q_stderr = None
        self._stdout = None
        self._stderr = None
        self._always_log_stderr = False
        self._working_directory=working_directory

    def activate_live_logger(self):
        self._live_logger = True

    def nonblocking_run(self):
        additional_env_commands, use_shell = self._setup_environment()
        if use_shell:
            all_commands = additional_env_commands + " ".join(self._commands)
        else:
            all_commands = additional_env_commands + self._commands
        self._process = subprocess.Popen(all_commands, stdout=subprocess.PIPE, 
                                         shell=use_shell, 
                                         stderr=subprocess.PIPE, text=True, 
                                         cwd=self._working_directory,
                                         errors='replace')

    def run(self):
        import concurrent
        with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self._run)
            res = future.result()
        return res
        
    def _run(self):
        self.nonblocking_run()
        if self._live_logger:
            self._stdout, self._stderr = self.live_logging_output()
        else:
            self._stdout, self._stderr = self.blocking_get_output()
        self.log_stderr()
        return self._stdout, self._stderr, self._process.returncode

    def live_logging_output(self):
        stdout = stderr = ""      
        for out_line, err_line in self.read_popen_pipes():
            if out_line:
                logger.info(self._modify_output(out_line))
                stdout += out_line
            if err_line:
                logger.error(self._modify_output(err_line))
                stderr += err_line

            if self.is_process_done():
                break
        return stdout, stderr
    
    def blocking_get_output(self):
        stdout, stderr = self._process.communicate()
        logger.debug(stdout)
        logger.debug(stderr)

        return stdout, stderr

    def log_stderr(self):
        bad_exit = self._process.returncode != 0 
        stderr_not_empty = self._stderr != ""
        log_errors = bad_exit or (stderr_not_empty and self._always_log_stderr)
        if log_errors:
            if self._process.returncode !=0:
                logger.error(self._stderr)
                self._log_errors_for_failed_commands()
            else:
                logger.warning(self._stderr)

    @staticmethod
    def _modify_output(output):
        return output.strip("\n")

    def _log_errors_for_failed_commands(self):
        command_str = " ".join([str(x) for x in self._commands])
        file_string = "\nCheck executable results and log files in:\n"+os.getcwd()
        logger.error("The executable with the following commands failed:\n"+command_str+file_string)

    def read_popen_pipes(self):
        with ThreadPoolExecutor(2) as pool:
            self._q_stdout, self._q_stderr = Queue(), Queue()
            pool.submit(self._enqueue_output, self._process.stdout, self._q_stdout)
            pool.submit(self._enqueue_output, self._process.stderr, self._q_stderr)
            while True:
                if self.is_process_done() and self._is_all_output_empty():
                    break
                out_line = err_line = ""

                try:
                    out_line = self._q_stdout.get_nowait()
                except Empty:
                    pass

                try:
                    err_line = self._q_stderr.get_nowait()
                except Empty:
                    pass

                yield (out_line, err_line)

    def is_process_done(self):
        return self._process.poll() is not None

    def _is_all_output_empty(self):
        return self._q_stdout.empty() and self._q_stderr.empty()

    @property
    def working_directory(self):
        return self._working_directory
    
    @staticmethod
    def _enqueue_output(working_file, queue):
       for line in iter(working_file.readline, ""):
            queue.put(line)
       working_file.close()

    def _setup_environment(self):
        setup_environment = MatCalExecutableEnvironmentSetupFunctionIdentifier.identify()
        additional_commands, use_shell = setup_environment(self._environment_commands)
        return additional_commands, use_shell


class ExternalExecutableCreatorBase(ObjectCreator):

    def _create_instance(self, runner, commands, modules_to_load, computer, 
                         working_directory):
        return runner(commands, modules_to_load, computer, working_directory)

    @abstractmethod
    def __call__(self, commands, modules_to_load, computer, working_directory):
        """Required method for building the ExternalExecutable object."""


class LocalExternalExecutable(ExternalExecutableBase):
    def __init__(self, commands, modules_to_load, computer=local_computer, 
                 working_directory=None):
        super().__init__(commands, modules_to_load, computer, working_directory)


class LocalExternalExecutableCreator(ExternalExecutableCreatorBase):
    def __init__(self):
        super().__init__()

    def __call__(self, commands, modules_to_load, computer=local_computer, 
                 working_directory=None):
        return self._create_instance(LocalExternalExecutable, commands, modules_to_load, 
                                     computer=computer, working_directory=working_directory)


class RemoteExternalExecutable(ExternalExecutableBase):
    def __init__(self, commands, modules_to_load, computer, working_directory=None):
        super().__init__(commands, modules_to_load, computer, working_directory=working_directory)

    def run(self):
        env_setup = MatCalPlatformEnvironmentSetupIdentifier.identify()
        env_setup.prepare()
        results = super().run()
        env_setup.reset()
        return results


class RemoteExternalExecutableCreator(ExternalExecutableCreatorBase):

    def __call__(self, commands, modules_to_load, computer, working_directory):
        return self._create_instance(RemoteExternalExecutable, commands, 
                                     modules_to_load, computer, working_directory)


class ExternalExecutableFactory(SpecificObjectFactory):

    class ExternalExecutableFactoryTypeError(RuntimeError):
        def __init__(self, *args, **kargs):
            super().__init__(*args, **kargs)

    def create(self, commands, modules_to_load, computer=local_computer, 
               working_directory=None):
        try:
            return super().create(type(computer), commands=commands, 
                                  modules_to_load=modules_to_load,
                                   computer=computer, working_directory=working_directory)
        except ValueError:
            from inspect import getmro
            return super().create(getmro(type(computer))[1], commands=commands, 
                                  modules_to_load=modules_to_load,
                                  computer=computer, working_directory=working_directory)


class DefaultEnvCommandProcessorCreator(ObjectCreator):
        
    def __call__(self, *args, **kwargs):
        return default_environment_command_processor(*args, *kwargs)


class ListCommandError(RuntimeError):

    def __init__(self, command_type):
        message = f"Commands must be passed as a list. Passed as a {command_type}"
        super().__init__(message)


class ExecutableEnvironmentSetupBase(ABC):
    @abstractmethod
    def prepare(self):
        """Required method to setup the computing environment."""

    @abstractmethod
    def reset(self):
        """Required method to reset the computing environment."""


class ExecutableNoEnvironmentSetup(ExecutableEnvironmentSetupBase):

    def prepare(self):
        """"""

    def reset(self):
        """"""


MatCalPlatformEnvironmentSetupIdentifier = \
    IdentifierByTestFunction(ExecutableNoEnvironmentSetup())

MatCalExternalExecutableFactory = ExternalExecutableFactory()
MatCalExternalExecutableFactory.register_creator(LocalComputingPlatform, 
                                                 LocalExternalExecutableCreator())
MatCalExternalExecutableFactory.register_creator(RemoteComputingPlatform, 
                                                 RemoteExternalExecutableCreator())
MatCalExternalExecutableFactory.register_creator(HPCComputingPlatform, 
                                                 RemoteExternalExecutableCreator())

def default_environment_command_processor(commands:list):
    if not isinstance(commands, list):
        raise ListCommandError(type(commands))
    command_string = ";".join(commands)
    return command_string, True

MatCalExecutableEnvironmentSetupFunctionIdentifier = \
    IdentifierByTestFunction(default_environment_command_processor)


def attempt_to_execute(item_to_execute, max_attempts, pause_time, *item_args, 
                       **item_kwargs):
    current_attempt = 0
    had_successful_execution = False
    _attempt_to_execute_check_inputs(max_attempts)
    while current_attempt < max_attempts:
        current_attempt += 1
        try:
            item_to_execute(*item_args, **item_kwargs)
            had_successful_execution = True
            break
        except Exception as e:
            base_message = f"Failed to execute {item_to_execute}\n Possibly due to busy system.\n"
            message = _generate_attempt_failure_message(base_message, 
                                                        max_attempts, 
                                                        current_attempt, e)
            logger.info(message)
            sleep(pause_time)
    if not had_successful_execution:
        item_to_execute(*item_args, **item_kwargs)
        had_successful_execution = True
    return had_successful_execution, current_attempt


def _attempt_to_execute_check_inputs(max_attempts):
    if max_attempts < 1:
        raise NonPositiveIntegerError(max_attempts)


class NonPositiveIntegerError(RuntimeError):
    
    def __init__(self, value):
        message = f"Positive integer required, passed {value}"
        super().__init__(message)


def _generate_attempt_failure_message(base_message, max_load_attempts,
                                      current_attempt, e):
    message = base_message
    if current_attempt <= max_load_attempts:
        message += f"Pausing then attempting again."
        message += f" Attempt {current_attempt} of {max_load_attempts}"
    else:
        message += "Maximum attempts reached, Throwing Error"
        logger.log(f"{repr(e)}")
    return message


class SlurmHPCExecutableEnvironmentSetup(ExecutableEnvironmentSetupBase):
    def __init__(self):
        self._slurm_job_id = None
        self._slurm_jobid = None
        self._slurm_jobid_key = "SLURM_JOBID"
        self._slurm_job_id_key = "SLURM_JOB_ID"

    def prepare(self):
        if self._slurm_jobid_key in os.environ.keys():
            self._slurm_jobid = os.environ.pop(self._slurm_jobid_key)

        if self._slurm_job_id_key in os.environ.keys():
            self._slurm_job_id = os.environ.pop(self._slurm_job_id_key)

    def reset(self):
        if self._slurm_jobid is not None:
            os.environ[self._slurm_jobid_key]  = self._slurm_jobid

        if self._slurm_job_id is not None:
            os.environ[self._slurm_job_id_key] = self._slurm_job_id