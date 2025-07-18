from abc import abstractmethod
from collections import OrderedDict
import os

from matcal.core.best_material_file_writer import MatcalFileWriterFactory
from matcal.core.logger import initialize_matcal_logger
from matcal.core.serializer_wrapper import matcal_load
from matcal.core.study_base import StudyBase

from matcal.dakota.dakota_interfaces import DakotaOutputReader

logger = initialize_matcal_logger(__name__)


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
            
    def _run_study(self):
        if self._restart and self._restart_results_filename != None:
            self._results = matcal_load(self._restart_results_filename)
        self._updated_next_evaluation_id_number()
        input_file_string = self._prepare_dakota_input()
        matcal_callback_dict = self._select_matcal_interface()
        
        ########################################################################
        #                       Start Dakota Cluster
        # The cluster of code between the two comments needs to be kept at the
        # same level of scope in order for dakota to exit correctly.
        ########################################################################
        import dakota.environment as dakenv
        daklib = self._build_dakota_environment(input_file_string, matcal_callback_dict, dakenv)
        try:
            daklib.execute()
            del daklib
            del dakenv
        except Exception as e:
            del daklib
            del dakenv
            raise e
        import gc
        gc.collect()
        ########################################################################
        #                       End Dakota Cluster
        ########################################################################

        dakota_results = self._return_output_information("./dakota.out")
        self._initialize_parameters_if_skipped_by_dakota()
        self._results._set_outcome(self._package_results(dakota_results))
        
        return self._results

    def _select_matcal_interface(self):
        if self.__testing_fail:
            matcal_callback_dict = {'matcal_interface_batch':self._fail_matcal_evaluate_parameter_sets_batch}
        elif self._restart:
            matcal_callback_dict = {'matcal_interface_batch':self._restart_matcal_evaluate_parameter_sets_batch}
        else:
            matcal_callback_dict = {'matcal_interface_batch':self._fresh_matcal_evaluate_parameter_sets_batch}
        return matcal_callback_dict

    def _build_dakota_environment(self, input_file_string, matcal_callback_dict, dakenv):
        try:
            daklib = dakenv.study(callbacks=matcal_callback_dict, input_string=input_file_string)
        except Exception as e:
            logger.error("Failed to establish Dakota Environment. This is likely due to bad initialization parameters.")
            raise e
        return daklib

    def _fresh_matcal_evaluate_parameter_sets_batch(self, parameter_sets):
        return self._matcal_evaluate_parameter_sets_batch(parameter_sets, is_finite_difference_eval=False, is_restart=False)

    def _fail_matcal_evaluate_parameter_sets_batch(self, parameter_sets):
        batch_results = self._matcal_evaluate_parameter_sets_batch(parameter_sets, is_finite_difference_eval=False, is_restart=False)
        raise RuntimeError("Intentional Testing Faulure-- Envoked:_for_testing_fail_after_first_batch")

    def _restart_matcal_evaluate_parameter_sets_batch(self, parameter_sets):
        return self._matcal_evaluate_parameter_sets_batch(parameter_sets, is_finite_difference_eval=False, is_restart=True)

    def _initialize_parameters_if_skipped_by_dakota(self):
        if len(self._results.parameter_history) < 1: 
            logger.warning("Dakota skipped evaluations, May be reusing old results. Please consult Dakota Files")
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
    

class DakotaCalibrationStudyBase(DakotaStudyBase):
    """
    Not intended for users: base class for all Dakota calibration studies.
    """

    study_class = "calibration"
    _best_material_filename = "best_material.inc"

    def _return_output_information(self, output_filename):
        best_parameters = self._dakota_reader(output_filename).parse_calibration()

        return best_parameters

    def _package_results(self, dakota_results):
        """"""
        return dakota_results


    def _study_specific_postprocessing(self):
        self._make_best_material_file()

    def _make_best_material_file(self):
        for eval_set in self._evaluation_sets.values():
            self._create_and_write_best_material_file(eval_set.model)

    def _create_and_write_best_material_file(self, model):
        file_writer = MatcalFileWriterFactory.create(model.executable, self._results.outcome)
        filename = self._make_best_filename(model)
        file_writer.write(filename)

    def _make_best_filename(self, model):
        return "Best_Material_{}.inc".format(model.name)

    def launch(self):
        """
        Dakota calibration studies return calibration information which include 
        the best fit parameter set, the objective or residuals for that parameter
        set and the evaluation ID for the parameter set.

        This information is stored in a dictionary with the keys "parameters", 
        "objectives", and "evaluation IDs". 

        The value for results["parameters"] is a dictionary of parameter values and 
        names for the best parameter set.

        The value for results["objectives"] is a list of objectives for the best evaluation 
        from the study where the objectives are floats.

        The value for results["evaluation IDs"] is an integer corresponding 
        to the evaluation number for the best parameter set.

        If the study returns multiple best parameter sets, such as with a 
        :class:`~matcal.dakota.global_calibration_studies.MultiObjectiveGACalibrationStudy`,
        the study will return a list of each of the above, where the indices of the lists match
        the best parameter results together.
        """
        return super().launch()

    


