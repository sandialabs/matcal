"""
The simulator module contains classes that launch models and
process results to be passed back to MatCal after being run. 
The only user facing class is the SimulatorResults class.
"""

import os
from abc import ABC, abstractmethod

from matcal.core.constants import DESIGN_PARAMETER_FILE, STATE_PARAMETER_FILE
from matcal.core.data import convert_dictionary_to_data
from matcal.core.external_executable import MatCalExternalExecutableFactory
from matcal.core.file_modifications import process_template_file
from matcal.core.logger import initialize_matcal_logger
from matcal.core.parameters import get_parameters_according_to_precedence
from matcal.core.object_factory import DefaultObjectFactory, ObjectCreator
from matcal.core.reporter import MatCalParameterReporterIdentifier
from matcal.core.serializer_wrapper import matcal_save
from matcal.core.utilities import matcal_name_format


logger = initialize_matcal_logger(__name__)


import contextlib
@contextlib.contextmanager
def string_out_err_capture():
    import sys
    from io import StringIO
    oldout,olderr = sys.stdout, sys.stderr
    try:
        out=[StringIO(), StringIO()]
        sys.stdout,sys.stderr = out
        yield out
    finally:
        sys.stdout,sys.stderr = oldout, olderr
        out[0] = out[0].getvalue()
        out[1] = out[1].getvalue()
    return out


class SimulatorResults:
    """Results data structure returned from a simulator."""
    def __init__(self, results_data, stdout, stderr, return_code, source_filename=None):
        self._results_data = results_data
        self._stdout = stdout
        self._stderr = stderr
        self._return_code = return_code
        self._source_filename = source_filename

    @property
    def results_data(self):
        """
        Returns the simulation results or None. None is returned if no 
        results are expected.

        :return: simulation results data
        :rtype: :class:`~matcal.core.data.Data`, None
        """
        return self._results_data

    @property
    def stdout(self):
        """
        :return: simulation execution standard output
        :rtype: str
        """
        return self._stdout

    @property
    def stderr(self):
        """
        :return: simulation execution error output
        :rtype: str
        """
        return self._stderr

    @property
    def return_code(self):
        """
        :return: simulation return code
        :rtype: int
        """
        return self._return_code

    @property
    def source_filename(self):
        """
        :return: Relevant source file for data
        :rtype: str
        """
        return self._source_filename


class SimulatorFailureResults(SimulatorResults):

    def __init__(self, stdout, stderr, return_code, state):
        self._state = state
        super().__init__(None, stdout, stderr, return_code)

    @property
    def state(self):
        return(self._state)


class Simulator(ABC):
    """
    Not intended for users: base class for simulators which run models.
    """
    def __init__(self,
                 model_name,
                 compute_information,
                 results_information,
                 state,
                 template_dir='.'):
        self._model_name = model_name
        self._state = state
        self._compute_information = compute_information
        self._results_information = results_information
        self._template_dir = template_dir
        self._commands = None
        self._modules_to_load = None
        self._initial_working_dir = os.getcwd()

    @abstractmethod
    def run(self, parameters, working_dir=None) -> SimulatorResults:
        """"""

    def _select_working_dir(self, working_dir):
        model_state_dir = os.path.join(matcal_name_format(self.model_name), 
                                       matcal_name_format(self._state.name))
        if working_dir is None:
            working_dir = os.path.abspath(os.path.join(os.getcwd(), model_state_dir))
        else:
            working_dir = os.path.abspath(os.path.join(working_dir, model_state_dir))

        logger.debug("Simulator working directory: {}".format(working_dir))
        return working_dir

    def _change_to_working_dir(self, working_dir=None):
        self._initial_working_dir = os.getcwd()
        os.chdir(self._select_working_dir(working_dir))

    @property
    def results_filename(self):
        return self._results_information.results_filename
    
    @property
    def model_name(self):
        return self._model_name

    @property
    def commands(self):
        return self._commands

    @property
    def number_of_cores(self):
        return self._compute_information.number_of_cores

    @property
    def state(self):
        return self._state

    @property
    def computer(self):
        return self._compute_information.computer

    @abstractmethod
    def get_results(self):
        """"""
    
    @property
    def fail_calibration_on_simulation_failure(self):
        return self._compute_information.fail_on_simulation_failure


class ExecutableSimulator(Simulator):
    """
    Not intended for users: runs models that require and external executable.
    """

    def __init__(self,
                 model_name,
                 compute_information,
                 results_information,
                 state, model_constants,
                   template_dir='.', 
                 commands=[]):
        
        super().__init__(model_name,
                    compute_information,
                    results_information,  
                    state,
                    template_dir=template_dir)
        self._commands = list(commands)
        self._model_constants = model_constants

    def run(self, parameters, working_dir=None, get_results=True):
        workdir_full_path = self._select_working_dir(working_dir)
        external_executable = MatCalExternalExecutableFactory.create(self._commands, 
            self._modules_to_load, self._compute_information.computer, 
            working_directory=workdir_full_path)
        model_params = get_parameters_according_to_precedence(self.state, 
                                                              self._model_constants, 
                                                              parameters)
        self._pass_parameters_to_simulators(workdir_full_path, model_params)
        self._write_parameters_file(workdir_full_path, parameters)
        results = None
        stdout = None
        stderr = None
        return_code = None
        try:
            stdout, stderr, return_code = self._execute_external(external_executable)
        except Exception as e:
            if self.fail_calibration_on_simulation_failure:
                error_str = (f"State \"{self.state.name}\" for model \"{self.model_name}\" "+
                            f"failed with error:\n{repr(e)}"+
                            " Usually, this is caused by an incorrect executable name. Exiting.")
                raise RuntimeError(error_str)
            else:
                    logger.error(f"Continuing after state \"{self.state.name}\" for "+
                                 f"model \"{self.model_name}\" "+
                                 f"failed with error:\n{repr(e)}.")
        if (return_code is not None and return_code != 0 and 
                self.fail_calibration_on_simulation_failure):
            error_str = (f"State \"{self.state.name}\" for model \"{self.model_name}\" "+
                         f"failed with exit code {return_code}. "+
                         f"Exiting. The following error was output from the executable:\n{stderr}")
            raise RuntimeError(error_str)
        if get_results:
            results = self._gather_results(workdir_full_path, stdout, stderr, return_code)
        else:
            results = SimulatorResults(None, stdout, stderr, return_code, None)
        return results

    def _gather_results(self, workdir_full_path, stdout, stderr, return_code):
        model_results, read_error = self.get_results(workdir_full_path)
        if model_results is not None and read_error is None:
            results_file = os.path.join(workdir_full_path, 
                    self._results_information.results_filename)
            results = SimulatorResults(model_results, stdout, stderr, return_code, results_file)
        elif not self.fail_calibration_on_simulation_failure and model_results is None:
            results = SimulatorFailureResults(stdout, stderr, return_code, self.state)
        else:
            raise read_error
        return results

    def _pass_parameters_to_simulators(self, workdir_full_path, parameters):
        files = [f.path for f in os.scandir(workdir_full_path)]
        for file in files:
            logger.debug(f"\t\tPreprocessing file {os.path.basename(file)}")
            if os.path.basename(file) not in [STATE_PARAMETER_FILE, DESIGN_PARAMETER_FILE]:
                process_template_file(file, parameters)

    def _write_parameters_file(self, workdir_full_path, parameters):
        params_file = os.path.join(workdir_full_path, DESIGN_PARAMETER_FILE)
        dictionary_reporter = MatCalParameterReporterIdentifier.identify()
        dictionary_reporter(params_file, parameters)

    def _execute_external(self, external_executable):
        logger.debug("Executing external application...")
        stdout, stderr, return_code = external_executable.run()
        simulation_out = "simulation.out"
        simulation_err = "simulation.err"
        if external_executable.working_directory:
            simulation_out = os.path.join(external_executable.working_directory,"simulation.out")
            simulation_err = os.path.join(external_executable.working_directory,"simulation.err")

        with open(simulation_out, "w") as fout:
            fout.write(stdout)

        with open(simulation_err, "w") as ferr:
            ferr.write(stderr)
        logger.debug("Simulator running complete! \n")

        return stdout, stderr, return_code

    def get_results(self, working_dir=None):
        data_filename = os.path.join(working_dir, self._results_information.results_filename)
        results = None
        error = None
        try:
            results = self._results_information.read(data_filename)
            results.set_state(self._state)
        except Exception as e:
            logger.error(f"Cannot read results file "+
                         f"with name \"{os.path.basename(data_filename)}\" for "+
                          f"model \"{self.model_name}\" and state \"{self.state.name}\". "+
                          "Caught the following "
                          f"error:\n{repr(e)}")    
            error = e
        return results, error


class PythonSimulator(Simulator):
    """
    Not intended for users: Runs python models.
    """

    def __init__(self, name, compute_information, results_information, state, model, field_coordinates=None):
        super().__init__(name, compute_information, results_information, state)
        self._workdir = None
        self._orig_stdout = None
        self._orig_stderr = None
        self._model = model
        self._field_coordinates = field_coordinates
        self._archive_name = None
        self._save_dir = "matcal_python_results_archive"
        if not os.path.exists(self._save_dir):
            os.mkdir(self._save_dir)

    def get_results(self):
        pass

    def run(self, parameters, working_dir=None):
        output = None
        try:
            logger.debug("Python Simulation: Running")
            with string_out_err_capture() as out:
                results = self._run_python_simulation(parameters, working_dir)
                self._archive_results(results, parameters)
            logger.debug("Python Simulation: Finished")
        except Exception as e:
            logger.error("Python Model \"{}\" Error: {}".format(self.model_name, repr(e)))
            if self.fail_calibration_on_simulation_failure:
                raise e
            else:
                logger.error("Continuing Study After Model \"{}\" Error.".format(self.model_name, 
                                                                                 repr(e)))
                stdout, stderr = self._extract_out_and_err(out)
                output = SimulatorFailureResults(stdout, stderr, None, self._state)

        if output == None:
            stdout, stderr = self._extract_out_and_err(out)
            output = SimulatorResults(results, stdout, stderr, None, self._archive_name)
        return  output

    def _extract_out_and_err(self, out):
        try:
            stdout, stderr = out 
        except AttributeError:
            stdout = "doc building no output catch"
            stderr = ""
        return stdout,stderr

    def _archive_results(self, results, parameters):
        self._archive_name = os.path.join(self._save_dir, self.model_name)
        for name, value in parameters.items():
            self._archive_name+= f"_{name}={value}"
        self._archive_name += ".joblib"
        matcal_save(self._archive_name, results)

    def _run_python_simulation(self, parameters, working_dir):
        run_variables = {**self._state.params}
        run_variables.update(self._model.get_model_constants(self._state))
        run_variables.update(parameters)
        logger.debug("{}".format(run_variables))

        results = self._python_function(**run_variables)
        results = self._convert_to_data(results)
        results.set_state(self._state)
        return results

    def _convert_to_data(self, results):
        converter = MatCalDataReaderFactory.create(self._is_field_simulation(),
            self._field_coordinates)
        results = converter(results)
        return results

    def _is_field_simulation(self):
        return self._field_coordinates != None

    def _python_function(self, **run_variables):
        return self._model.python_function(**run_variables)

    @property
    def _results_file_path(self):
        return None

    @property
    def results_filename(self):
        return self._archive_name


class ProbeDataReaderCreator(ObjectCreator):

    def __call__(self, *args, **kwargs):
        return convert_dictionary_to_data

    
class DataReaderFactory(DefaultObjectFactory):
    pass


MatCalDataReaderFactory = DataReaderFactory(ProbeDataReaderCreator())
