import os
import time
from collections import OrderedDict
from unittest import mock

import numpy as np

from matcal.core.constants import EVALUATION_EXTENSION, IN_PROGRESS_RESULTS_FILENAME
from matcal.core.parameters import Parameter, ParameterCollection
from matcal.core.restart_file import BatchRestartNone
from matcal.core.study_base import StudyResults
from matcal.core.tests.MatcalUnitTest import MatcalUnitTest
from matcal.core.tests.unit.test_study_base import StudyBaseUnitTests
from matcal.dakota import dakota_studies
from matcal.dakota.dakota_studies import (
    _flush_library,
    _parameters_match,
    _read_text_if_modified_since,
    _tee_process_stdout,
)
from matcal.dakota.local_calibration_studies import (
    CobylaCalibrationStudy,
)
from matcal.dakota.local_calibration_studies import (
    GradientCalibrationStudy as _MinimalCalibrationStudy,
)


class TestDakotaStudyBase:
    def __init__():
        pass
    class CommonTests(StudyBaseUnitTests.CommonTests):
        
        def _write_fake_restart_files(self, restart_filename="dakota.rst", 
                                      matcal_results=
                                      IN_PROGRESS_RESULTS_FILENAME+"."+EVALUATION_EXTENSION):
            with open(restart_filename, "w") as f:
                f.write("\n")
            with open(matcal_results, "w") as f:
                f.write("\n")
        def test_set_restart_file_not_found(self):
            study = self._study_class(self.parameter_collection)
            self._set_study_specific_options(study)

            with self.assertRaises(FileNotFoundError):
                study.restart()
       
        def test_write_restart_filename(self):
            study = self._study_class(self.parameter_collection)
            study.add_evaluation_set(self.mock_model, self.objective, self.gold_results)
            self._set_study_specific_options(study)
            with self.assertRaises(TypeError):
                study.set_restart_filename(1)

            goal_fn = "my_filename.rst"
            study.set_restart_filename(goal_fn)
            self.assertEqual(goal_fn, study.get_write_restart_filename().strip("\""))
            study._initialize_study_and_batch_evaluator() 
            study._prepare_dakota_input() 
            input_file_string = study.get_input_string()
            self.assertTrue(f"write_restart = \"{goal_fn}\"" in input_file_string)
            self.assertFalse(f"read_restart = " in input_file_string)

        def test_restart_custom_filename(self):
            study = self._study_class(self.parameter_collection)
            study.add_evaluation_set(self.mock_model, self.objective, self.gold_results)
            self._set_study_specific_options(study)

            custom_filename = "my_filename.rst"
            self._write_fake_restart_files(custom_filename)
            study.set_restart_filename(custom_filename)
            study.restart(custom_filename)
            abs_path_filename = os.path.join(os.getcwd(), custom_filename)
            
            self.assertEqual(custom_filename, study.get_write_restart_filename().strip("\""))

            study._initialize_study_and_batch_evaluator()
            study._prepare_dakota_input()    
            input_file_string = study.get_input_string()
            self.assertTrue(f"write_restart = \"{custom_filename}\"" in input_file_string)
            self.assertTrue(f"read_restart = \"{abs_path_filename}\"" in input_file_string)

        def test_restart_custom_filename_write_only(self):
            study = self._study_class(self.parameter_collection)
            study.add_evaluation_set(self.mock_model, self.objective, self.gold_results)
            self._set_study_specific_options(study)

            custom_filename = "my_filename.rst"
            self._write_fake_restart_files(custom_filename)
            study.set_restart_filename(custom_filename)
            abs_path_filename = os.path.join(os.getcwd(), custom_filename)
            
            self.assertEqual(custom_filename, study.get_write_restart_filename().strip("\""))

            study._initialize_study_and_batch_evaluator()
            study._prepare_dakota_input()    
            input_file_string = study.get_input_string()
            self.assertTrue(f"write_restart = \"{custom_filename}\"" in input_file_string)
            self.assertFalse(f"read_restart = " in input_file_string)

        def test_restart(self):
            study = self._study_class(self.parameter_collection)
            study.add_evaluation_set(self.mock_model, self.objective, self.gold_results)
            self._set_study_specific_options(study)
            self._write_fake_restart_files()
            study.restart()
            self.assertTrue(study._restart)
            abs_path_filename = os.path.join(os.getcwd(), "dakota.rst")
            self.assertEqual(study.get_read_restart_filename().strip("\""), abs_path_filename)
            study._initialize_study_and_batch_evaluator()
            study._prepare_dakota_input()    
            input_file_string = study.get_input_string()
            self.assertTrue(f"read_restart = \"{abs_path_filename}\"" in input_file_string)

        def test_set_verbosity(self):
            study = self._study_class(self.parameter_collection)
            study.add_evaluation_set(self.mock_model, self.objective, self.gold_results)
            self._set_study_specific_options(study)
            with self.assertRaises(ValueError):
                study.set_output_verbosity("not valid")
            study.set_output_verbosity("verbose")
            study._prepare_dakota_input()
            study._initialize_study_and_batch_evaluator()    
            input_file_string = study.get_input_string()
            self.assertTrue(f"output = verbose" in input_file_string)

        def test_format_parameters(self):
            study = self._study_class(self.parameter_collection)
            unprocessed_params_input = {"functions":1, "variables":1, "cv":[0], "cv_labels":["a"]}
            processed_params = study._format_parameters(unprocessed_params_input)
            self.assertEqual({"a":0}, processed_params)

        def test_convert_results_list_for_dakota(self):
            results = [np.zeros((1,6)), np.ones((1,6))]
            study = self._study_class(self.parameter_collection)
            converted_results = study._convert_results_list_for_dakota(results)
            for idx, converted_result in enumerate(converted_results):
                self.assertTrue("fns" in converted_result)
                self.assert_close_arrays(converted_result["fns"], results[idx])

        def test_matcal_evaluate_parameter_sets_batch(self):
            study = self._study_class(self.parameter_collection)
            study.add_evaluation_set(self.mock_model, self.objective, self.gold_results)
            study._parameter_batch_evaluator = study._initialize_study_and_batch_evaluator()
            study._restart_obj = BatchRestartNone(None, None)
            unprocessed_params_input = {"functions":6, "variables":1, "cv":[0.0], "cv_labels":["a"]}

            results = study._matcal_evaluate_parameter_sets_batch(unprocessed_params_input)
            if isinstance(results, list):
                results = results[0]
            goal =  {'fns':np.zeros((1,6))}
            self.assert_close_arrays(results['fns'],goal['fns'])

class TestTeeProcessStdout(MatcalUnitTest):
    def setUp(self):
        super().setUp(__file__)

    def test_os_level_writes_are_captured_in_file(self):
        with _tee_process_stdout("captured.log"):
            os.write(1, b"fortran style output\n")
        with open("captured.log") as file_handle:
            self.assertIn("fortran style output", file_handle.read())

    def test_stdout_restored_after_exception(self):
        original_fd_target = os.fstat(1)
        with self.assertRaises(ValueError), _tee_process_stdout("captured.log"):
            raise ValueError("inside")
        self.assertEqual(os.fstat(1).st_ino, original_fd_target.st_ino)


class TestBestParameterHelpers(MatcalUnitTest):
    def setUp(self):
        super().setUp(__file__)

    def test_parameters_match_within_tolerance(self):
        self.assertTrue(_parameters_match({"a": 1.0, "b": 0.0}, {"a": 1.0 + 1e-12, "b": 0.0}))

    def test_parameters_do_not_match(self):
        self.assertFalse(_parameters_match({"a": 1.0}, {"a": 1.1}))

    def test_parameters_with_different_names_do_not_match(self):
        self.assertFalse(_parameters_match({"a": 1.0}, {"b": 1.0}))

    def test_read_text_if_modified_since_missing_file(self):
        self.assertEqual(_read_text_if_modified_since("missing.out", 0.0), "")

    def test_read_text_if_modified_since_old_file(self):
        with open("old.out", "w") as file_handle:
            file_handle.write("old")
        self.assertEqual(_read_text_if_modified_since("old.out", time.time() + 100), "")

    def test_read_text_if_modified_since_new_file(self):
        start_time = time.time() - 100
        with open("new.out", "w") as file_handle:
            file_handle.write("new")
        self.assertEqual(_read_text_if_modified_since("new.out", start_time), "new")


class TestNativeFlush(MatcalUnitTest):
    def setUp(self):
        super().setUp(__file__)

    def test_missing_library_is_ignored(self):
        _flush_library("library_that_does_not_exist_xyz", "fflush")

    def test_missing_function_is_ignored(self):
        _flush_library("c", "function_that_does_not_exist_xyz")

    def test_flush_existing_function(self):
        _flush_library("c", "fflush")


class TestTeeChildHoldsStdout(MatcalUnitTest):
    def setUp(self):
        super().setUp(__file__)

    def test_warns_when_output_is_still_held(self):
        with mock.patch.object(dakota_studies, "_TEE_JOIN_TIMEOUT_SECONDS", 0.1), \
             mock.patch.object(dakota_studies.logger, "warning") as warning:
            with _tee_process_stdout("captured.log"):
                held_fd = os.dup(1)
            os.close(held_fd)
        warning.assert_called_once()


def _calibration_study_with_history(total_objectives, parameter_values):
    study = _MinimalCalibrationStudy(
        ParameterCollection("p", Parameter("a", 0, 10, 1.0)))
    study._results = StudyResults()
    study._results._total_objective_history = list(total_objectives)
    study._results._parameter_history = OrderedDict(a=list(parameter_values))
    return study


class TestCalibrationBestParameters(MatcalUnitTest):
    def setUp(self):
        super().setUp(__file__)

    def test_matcal_best_parameters_none_without_history(self):
        study = _calibration_study_with_history([], [])
        self.assertIsNone(study._matcal_best_parameters())

    def test_matcal_best_parameters_uses_lowest_objective(self):
        study = _calibration_study_with_history([3.0, 1.0, 2.0], [5.0, 6.0, 7.0])
        self.assertEqual(study._matcal_best_parameters(), {"best:a": 6.0})

    def test_warns_when_dakota_best_differs(self):
        study = _calibration_study_with_history([3.0, 1.0], [5.0, 6.0])
        with mock.patch.object(dakota_studies.logger, "warning") as warning:
            study._package_results(OrderedDict({"best:a": 5.0}))
        warning.assert_called_once()

    def test_no_warning_when_dakota_best_agrees(self):
        study = _calibration_study_with_history([3.0, 1.0], [5.0, 6.0])
        with mock.patch.object(dakota_studies.logger, "warning") as warning:
            study._package_results(OrderedDict({"best:a": 6.0}))
        warning.assert_not_called()

    def test_output_information_falls_back_to_text_reader(self):
        study = _calibration_study_with_history([], [])
        support_dir = os.path.join(os.path.dirname(__file__), "..", "test_support_files")
        results = study._return_output_information(os.path.join(support_dir, "dakota.out"))
        self.assertAlmostEqual(results["best:Y"], 500.49999998)

    def test_output_information_prefers_final_dakota_variables(self):
        study = _calibration_study_with_history([], [])
        study._final_dakota_parameters = OrderedDict(a=2.5)
        self.assertEqual(study._return_output_information("unused"), {"best:a": 2.5})


class _FakeDakotaEnvironment:
    @staticmethod
    def get_variable_values(dakota_study):
        return [4.5]


class TestCaptureFinalVariables(MatcalUnitTest):
    def setUp(self):
        super().setUp(__file__)

    def test_single_best_point_study_reads_final_variables(self):
        study = _calibration_study_with_history([], [])
        study._capture_final_variables(_FakeDakotaEnvironment, None)
        self.assertEqual(study._final_dakota_parameters, {"a": 4.5})

    def test_multi_point_study_keeps_text_reader(self):
        study = CobylaCalibrationStudy(ParameterCollection("p", Parameter("a", 0, 10, 1.0)))
        study._capture_final_variables(_FakeDakotaEnvironment, None)
        self.assertIsNone(study._final_dakota_parameters)

    def test_multi_point_study_does_not_check_best(self):
        study = CobylaCalibrationStudy(ParameterCollection("p", Parameter("a", 0, 10, 1.0)))
        with mock.patch.object(study, "_warn_if_best_differs_from_matcal") as check:
            study._package_results(OrderedDict({"best:a": 5.0}))
        check.assert_not_called()
