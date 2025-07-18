import os
import numpy as np
from matcal.core.constants import (IN_PROGRESS_RESULTS_FILENAME, EVALUATION_EXTENSION)
from matcal.core.tests.unit.test_study_base import StudyBaseUnitTests


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
            unprocessed_params_input = {"functions":6, "variables":1, "cv":[0.0], "cv_labels":["a"]}

            results = study._matcal_evaluate_parameter_sets_batch(unprocessed_params_input)
            if isinstance(results, list):
                results = results[0]
            goal =  {'fns':np.zeros((1,6))}
            self.assert_close_arrays(results['fns'],goal['fns'])