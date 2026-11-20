import ctypes
import ctypes.util
import multiprocessing
import multiprocessing.forkserver
import os
import sys
import threading
from abc import abstractmethod
from collections import OrderedDict
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from typing import BinaryIO

from matcal.core.best_material_file_writer import matcal_file_writer_factory
from matcal.core.logger import initialize_matcal_logger
from matcal.core.serializer_wrapper import matcal_load
from matcal.core.study_base import StudyBase
from matcal.dakota.dakota_interfaces import DakotaOutputReader, DakotaTermination
from matcal.dakota.gradient_convergence import (
    MatCalConvergenceMonitor,
    MatCalConvergenceReached,
    batch_center,
    scale_to_bounds,
    total_objective,
)

logger = initialize_matcal_logger(__name__)

DAKOTA_STDOUT_FILENAME = "dakota_stdout.log"
OPTPP_OUTPUT_FILENAME = "OPT_DEFAULT.out"
_TEE_JOIN_TIMEOUT_SECONDS = 10.0
_TEE_CHUNK_SIZE = 65536


def _write_all(file_descriptor: int, data: bytes) -> None:
    """Write all of ``data``; ``os.write`` may write only part of it."""
    while data:
        data = data[os.write(file_descriptor, data):]


def _try_write(write: Callable[..., object], *arguments: object) -> bool:
    # Logging here would write to the captured stream itself, so failures
    # are only reported through the return value.
    try:
        write(*arguments)
    except (OSError, ValueError):
        return False
    return True


def _close_quietly(file_descriptor: int) -> None:
    try:
        os.close(file_descriptor)
    except OSError:
        pass


def _copy_stream(read_fd: int, output_fd: int, log_file: BinaryIO) -> None:
    """
    Copy the pipe to the original standard output and the log until the pipe
    closes. A destination that fails is dropped, but the pipe is always
    drained so the solver can never block on a full pipe. The reader owns
    ``output_fd`` and closes it when done.
    """
    terminal_ok, log_ok = True, True
    chunk = os.read(read_fd, _TEE_CHUNK_SIZE)
    while chunk:
        terminal_ok = terminal_ok and _try_write(_write_all, output_fd, chunk)
        log_ok = log_ok and _try_write(log_file.write, chunk)
        chunk = os.read(read_fd, _TEE_CHUNK_SIZE)
    os.close(read_fd)
    _close_quietly(output_fd)


def _start_forkserver_if_used() -> None:
    # A forkserver started inside the tee would inherit the pipe and keep it open.
    if multiprocessing.get_start_method() == "forkserver":
        multiprocessing.forkserver.ensure_running()


def _flush_library(library_name: str, flush_function_name: str) -> None:
    library_path = ctypes.util.find_library(library_name)
    if library_path is None:
        return
    flush_function = getattr(ctypes.CDLL(library_path), flush_function_name, None)
    if flush_function is not None:
        flush_function(None)


def _flush_native_stdout() -> None:
    # Compiled solvers buffer standard output when it is not a terminal.
    # A NULL argument flushes all C streams and all gfortran units.
    _flush_library("c", "fflush")
    _flush_library("gfortran", "_gfortran_flush_i4")


def _restore_stdout(saved_fd: int, reader: threading.Thread) -> None:
    sys.stdout.flush()
    _flush_native_stdout()
    os.dup2(saved_fd, 1)
    reader.join(timeout=_TEE_JOIN_TIMEOUT_SECONDS)
    if reader.is_alive():
        # The reader still writes to saved_fd and closes it when it ends.
        logger.warning("Captured solver output may be incomplete: a child "
                       "process still holds standard output.")


@contextmanager
def _tee_process_stdout(log_filename: str) -> Iterator[None]:
    """
    Copy everything written to the process standard output file descriptor,
    including output from compiled solvers, to ``log_filename`` while still
    passing it through to the original standard output.

    :param log_filename: the file that receives a copy of the output.
    """
    _start_forkserver_if_used()
    sys.stdout.flush()
    saved_fd = os.dup(1)
    read_fd, write_fd = os.pipe()
    os.dup2(write_fd, 1)
    os.close(write_fd)
    with open(log_filename, "wb") as log_file:
        reader = threading.Thread(target=_copy_stream, daemon=True,
                                  args=(read_fd, saved_fd, log_file))
        reader.start()
        try:
            yield
        finally:
            _restore_stdout(saved_fd, reader)


def _parameters_match(first: Mapping[str, float], second: Mapping[str, float],
                      relative_tolerance: float = 1e-8) -> bool:
    """
    :return: True if both mappings have the same names and values that agree
        to ``relative_tolerance``.
    """
    if set(first) != set(second):
        return False
    return all(abs(first[name] - second[name])
               <= relative_tolerance * max(abs(first[name]), abs(second[name]), 1.0)
               for name in first)


def _read_text_if_exists(filename: str) -> str:
    """:return: the file contents, or an empty string if it does not exist."""
    if not os.path.isfile(filename):
        return ""
    with open(filename, "r") as file_handle:
        return file_handle.read()


def _move_aside(filename: str) -> None:
    """
    Rename an existing ``filename`` to ``filename.previous``, replacing an
    older copy, so output read after a run cannot come from an earlier run.
    """
    if os.path.isfile(filename):
        os.replace(filename, f"{filename}.previous")


class DakotaStudyBase(StudyBase):

    @abstractmethod
    def _return_output_information(self, output_filename):
        """"""

    @abstractmethod
    def _package_results(self, dakota_results):
        """"""
        
    # from DakotaInputFile
    @abstractmethod
    def get_input_string(self):
        """"""

    # from DakotaInputFile
    @abstractmethod
    def write_input_file(self):
        """"""

    # from DakotaInputFile
    @abstractmethod
    def get_method_type_block(self):
        """"""
    
    # from DakotaInputFile
    @abstractmethod
    def set_read_restart_filename(self):
        """"""

    # from DakotaInputFile
    @property
    @abstractmethod
    def _needs_residuals():
        """"""

    # from DakotaInputFile
    def populate_variables():
        """"""

    # from DakotaInputFile
    def set_number_of_expected_responses():
        """"""

    def _for_testing_fail_after_first_batch(self):
        self.__testing_fail = True

    def __init__(self, *parameters):
        super().__init__(*parameters)
        self._dakota_reader = DakotaOutputReader
        self._restart = False
        self._restart_filename = None
        self._restart_results_filename = None
        self.__testing_fail = False
        self._convergence_monitor: MatCalConvergenceMonitor | None = None
        self._matcal_stop: DakotaTermination | None = None
            
    def _run_study(self):
        if self._restart and self._restart_results_filename != None:
            self._results = matcal_load(self._restart_results_filename)
        self._updated_next_evaluation_id_number()
        input_file_string = self._prepare_dakota_input()
        matcal_callback_dict = self._select_matcal_interface()
        self._before_dakota_run()

        ########################################################################
        #                       Start Dakota Cluster
        # The cluster of code between the two comments needs to be kept at the
        # same level of scope in order for dakota to exit correctly.
        ########################################################################
        import dakota.environment as dakenv
        daklib = self._build_dakota_environment(input_file_string, matcal_callback_dict, dakenv)
        try:
            with _tee_process_stdout(DAKOTA_STDOUT_FILENAME):
                daklib.execute()
            self._capture_final_variables(dakenv, daklib)
        except MatCalConvergenceReached as stop:
            # Raised by MatCal's own convergence check to stop Dakota early.
            self._handle_matcal_stop(stop.termination)
        finally:
            # Any other exception propagates after the references are released.
            del daklib
            del dakenv
        import gc
        gc.collect()
        ########################################################################
        #                       End Dakota Cluster
        ########################################################################

        dakota_results = self._return_output_information("./dakota.out")
        self._initialize_parameters_if_skipped_by_dakota()
        self._results._set_outcome(self._package_results(dakota_results))
        self._record_exit_information()
        return self._results

    def _before_dakota_run(self) -> None:
        """Hook to prepare the working directory just before Dakota runs."""

    def _capture_final_variables(self, dakota_environment, dakota_study) -> None:
        """Hook to read final results from the Dakota study before it is deleted."""

    def _handle_matcal_stop(self, termination: DakotaTermination) -> None:
        logger.info(f"MatCal stopped Dakota: {termination.message}")
        self._matcal_stop = termination

    def _record_exit_information(self) -> None:
        self._results._set_exit_information(None, None,
                                            "Termination information not recorded for Dakota." +
                                            "Check terminal output, 'dakota.out' and " +
                                            "other dakota related output files. ")

    def _select_matcal_interface(self):
        if self.__testing_fail:
            matcal_callback_dict = {'matcal_interface_batch':
                                    self._fail_matcal_evaluate_parameter_sets_batch}
        else:
            matcal_callback_dict = {'matcal_interface_batch':
                                    self._fresh_matcal_evaluate_parameter_sets_batch}
        return matcal_callback_dict

    def _build_dakota_environment(self, input_file_string, matcal_callback_dict, dakenv):
        try:
            daklib = dakenv.study(callbacks=matcal_callback_dict, input_string=input_file_string)
        except Exception as e:
            logger.error("Failed to establish Dakota Environment. "+
                         "This is likely due to bad initialization parameters.")
            raise e
        return daklib

    def _fresh_matcal_evaluate_parameter_sets_batch(self, parameter_sets):
        results = self._matcal_evaluate_parameter_sets_batch(parameter_sets, 
                                                             is_finite_difference_eval=False)
        self._check_matcal_convergence(parameter_sets, results)
        return results

    def _check_matcal_convergence(self, parameter_sets, dakota_results) -> None:
        if self._convergence_monitor is None:
            return
        center_functions = (dakota_results[0] if isinstance(dakota_results, list)
                            else dakota_results)["fns"]
        termination = self._convergence_monitor.update(
            scale_to_bounds(batch_center(parameter_sets), self._parameter_collection),
            total_objective(center_functions, self._needs_residuals),
            self._results.number_of_evaluations)
        if termination is not None:
            raise MatCalConvergenceReached(termination)

    def _fail_matcal_evaluate_parameter_sets_batch(self, parameter_sets):
        batch_results = self._matcal_evaluate_parameter_sets_batch(parameter_sets, 
                                                                   is_finite_difference_eval=False)
        raise RuntimeError("Intentional Testing Failure-- "+
                           "Invoked:_for_testing_fail_after_first_batch")

    def _initialize_parameters_if_skipped_by_dakota(self):
        if len(self._results.parameter_history) < 1: 
            logger.warning("Dakota skipped evaluations, May be reusing old results. "+
                           "Please consult Dakota Files")
            eval_params = {}
            for param_name in self._parameter_collection:
                eval_params[param_name] = "N/A"
            eval_name = "mock_eval"
            mock_batch_parameters = {eval_name:eval_params}
            self._results._update_parameter_history(mock_batch_parameters, [eval_name])

    def _prepare_dakota_input(self):
        self._set_residual_and_objective_sizes()
        self.populate_variables(self._parameter_collection)
        input_file_string = self.get_input_string()
        self.write_input_file()
        #input_file_string = get_string_from_text_file(DEFAULT_FILENAME)
        return input_file_string

    def _format_parameter_batch_eval_results(self, batch_raw_objectives, 
                                             flattened_batch_results, 
                                             total_objs, parameter_sets, batch_qois):
        combined_objs, combined_resids, eval_dirs = flattened_batch_results
        dakota_return_values = None
        if self._needs_residuals:
            logger.debug(" Dakota needs residual\n")
            dakota_return_values = combined_resids
        else:
            logger.debug(" Dakota needs objectives\n")
            dakota_return_values = combined_objs
        dakota_return_values = self._convert_results_list_for_dakota(dakota_return_values)
        if not isinstance(parameter_sets, list):
            dakota_return_values = dakota_return_values[-1]
        logger.debug(" Dakota return values length:\n")
        logger.debug(f"{len(dakota_return_values)}")
        return dakota_return_values

    def _convert_results_list_for_dakota(self, results):
        converted_results = []
        for result in results:
            converted_results.append({'fns':result})
        return converted_results

    def _format_parameters(self, parameter_set):
        num_fns = parameter_set["functions"]
        num_variables = parameter_set["variables"]
        cv = parameter_set["cv"]
        cv_labels = parameter_set["cv_labels"]
        new_param_dict = OrderedDict()
        for idx in range(num_variables):
            new_param_dict[cv_labels[idx]] = cv[idx]
        return new_param_dict

    def _set_residual_and_objective_sizes(self):
        number_of_residuals = 0
        number_of_objectives = 0
        for eval_set in self._evaluation_sets.values():
            number_of_objectives += eval_set.number_of_objectives
            number_of_residuals += eval_set.residual_vector_length
        self.set_number_of_expected_responses(number_of_objectives, 
                                                          number_of_residuals)
   
    def restart(self, restart_filename="dakota.rst",
                matcal_results_filename='in_progress_results.joblib'):
        """
        Allows the Dakota study to be restarted from a restart file.

        :param restart_filename: The Dakota restart filename to be used. This should be
            the filename relative to where the input file is and take into account
            any directory changes that the input file may have.
        :type restart_filename: str
        
        :param matcal_results_filename: The MatCal results file to be used. This should be
            the filename relative to where the input file is and take into account
            any directory changes that the input file may have.
        :type matcal_results_filename: str
        """
        self._restart = True

        if restart_filename != None:
            self.set_read_restart_filename(self._make_absolute_path(restart_filename))   
        if os.path.exists(matcal_results_filename): 
            self._restart_results_filename = self._make_absolute_path(matcal_results_filename)
    
    def _make_absolute_path(self, filename):
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Restart file \"{filename}\" not found. "
                                            "Cannot restart.")         
        else:
            filename = os.path.abspath(filename)
        return filename

    def _import_matcal_results(self, filename):
        self._results = matcal_load(filename)
    

def _package_best_parameters(parameters: Mapping[str, float]) -> OrderedDict:
    return OrderedDict((f"best:{name}", float(value)) for name, value in parameters.items())


class DakotaCalibrationStudyBase(DakotaStudyBase):
    """
    Not intended for users: base class for all Dakota calibration studies.
    """

    study_class = "calibration"
    _best_material_filename = "best_material.inc"
    # Studies with a single best point read it from Dakota and check it
    # against MatCal's history; multi-objective studies keep the text reader.
    _has_single_best_point: bool = False

    def __init__(self, *parameters):
        super().__init__(*parameters)
        self._final_dakota_parameters: OrderedDict | None = None

    def _return_output_information(self, output_filename):
        if self._final_dakota_parameters is not None:
            return _package_best_parameters(self._final_dakota_parameters)
        return self._dakota_reader(output_filename).parse_calibration()

    def _capture_final_variables(self, dakota_environment, dakota_study) -> None:
        if not self._has_single_best_point:
            return
        values = dakota_environment.get_variable_values(dakota_study)
        names = self._parameter_collection.get_item_names()
        self._final_dakota_parameters = OrderedDict(zip(names, values))

    def _package_results(self, dakota_results):
        """"""
        if self._has_single_best_point:
            self._warn_if_best_differs_from_matcal(dakota_results)
        return dakota_results

    def _matcal_best_values(self) -> OrderedDict | None:
        if len(self._results.total_objective_history) < 1:
            return None
        # Dakota's parameters, before any MatCal parameter preprocessors.
        history = self._results.parameter_history
        best_index = self._results.best_evaluation_index
        return OrderedDict((name, values[best_index])
                           for name, values in history.items())

    def _matcal_best_parameters(self) -> OrderedDict | None:
        best_values = self._matcal_best_values()
        return None if best_values is None else _package_best_parameters(best_values)

    def _handle_matcal_stop(self, termination: DakotaTermination) -> None:
        super()._handle_matcal_stop(termination)
        self._final_dakota_parameters = self._matcal_best_values()

    def _warn_if_best_differs_from_matcal(self, dakota_results: Mapping) -> None:
        matcal_best = self._matcal_best_parameters()
        if matcal_best is not None and not _parameters_match(dakota_results, matcal_best):
            logger.warning("Dakota's best parameters differ from MatCal's lowest total "
                           f"objective evaluation.\n Dakota: {dict(dakota_results)}"
                           f"\n MatCal: {dict(matcal_best)}")

    def _before_dakota_run(self) -> None:
        _move_aside(OPTPP_OUTPUT_FILENAME)

    def _record_exit_information(self) -> None:
        termination = (self._read_solver_termination()
                       if self._matcal_stop is None else self._matcal_stop)
        self._results._set_exit_information(termination.success, termination.status,
                                            termination.message)

    def _read_solver_termination(self) -> DakotaTermination:
        solver_texts = [_read_text_if_exists(filename)
                        for filename in (DAKOTA_STDOUT_FILENAME, OPTPP_OUTPUT_FILENAME)]
        method = self.get_method_type_block().name
        return self._dakota_reader("./dakota.out").parse_termination(method, solver_texts)

    def _study_specific_postprocessing(self):
        self._make_best_material_file()

    def _make_best_material_file(self):
        for eval_set in self._evaluation_sets.values():
            self._create_and_write_best_material_file(eval_set.model)

    def _create_and_write_best_material_file(self, model):
        file_writer = matcal_file_writer_factory.create(model.executable, self._results.outcome)
        filename = self._make_best_filename(model)
        file_writer.write(filename)

    def _make_best_filename(self, model):
        return "Best_Material_{}.inc".format(model.name)

    def launch(self):
        """
        Run the calibration and return its
        :class:`~matcal.core.study_base.StudyResults`.

        ``results.outcome`` holds the best parameter set found by Dakota with
        keys ``"best:<parameter name>"``, the same keys the SciPy calibration
        studies use. The best total objective, its evaluation and the full
        evaluation history are available from the results object, for example
        :meth:`~matcal.core.study_base.StudyResults.best_total_objective` and
        :meth:`~matcal.core.study_base.StudyResults.best_evaluation_index`.

        ``results.success``, ``results.exit_status`` and
        ``results.exit_message`` report the solver's termination reason when
        it can be found in the Dakota output, the solver's own output files or
        the standard output captured in ``dakota_stdout.log``. Otherwise
        ``success`` is None.

        If the study returns multiple best parameter sets, such as with a
        :class:`~matcal.dakota.global_calibration_studies.MultiObjectiveGACalibrationStudy`,
        the outcome values are lists whose indices match across parameters.
        """
        return super().launch()

    


