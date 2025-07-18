import os
import numpy as np
import random

from matcal.core.data import DataCollection, convert_dictionary_to_data
from matcal.core.input_file_writer import InputFileLine
from matcal.core.models import PythonModel
from matcal.core.objective import Objective, ObjectiveCollection
from matcal.core.parameters import ParameterCollection, Parameter
from matcal.core.tests.MatcalUnitTest import MatcalUnitTest
from matcal.core.utilities import get_current_files_path

from matcal.dakota.global_calibration_studies import (MultiObjectiveGACalibrationStudy, 
                                                      SingleObjectiveGACalibrationStudy)

from matcal.dakota.local_calibration_studies import (CobylaCalibrationStudy, 
                                                     GradientCalibrationStudy, 
                                                     MeshAdaptiveSearchCalibrationStudy, 
                                                     ParallelDirectSearchCalibrationStudy, 
                                                     PatternSearchCalibrationStudy, 
                                                     SolisWetsCalibrationStudy)



def parameter_preprocessor_func(params):
    for key in params:
        params[key] += 1
    params["new"] = 1.0
    params["new_str"] = "\"string\""
    params["new_str2"] = "\'string\'"
    params["new_str3"] = "string"

    return params

def my_model_func(a):
    return {'Y':a}

def my_model_func_2(a):
    return {'Y':a**2-2}

class DakotaMockStudyTestBase(object):
    def __init__():
        pass
    class CommonTests(MatcalUnitTest):
        def setUp(self, filename):
            super().setUp(filename)

            self._import_dir = "/".join(filename.split("/")[:-1])
            self.param = Parameter("a", 0, 10, 6.0+np.random.uniform(0, 0.001))
            self.PC = ParameterCollection("Test", self.param)
            self._input_file = "dakota.in"
            self._objective = Objective("Y")
            
            self._objectives = ObjectiveCollection("my_objectives", self._objective)
            self.tol = 1e-5
            data_dict = {"X":[1], "Y":[5]}
            fake_single_point_data = convert_dictionary_to_data(data_dict)
            self.DC = DataCollection("ONE", fake_single_point_data)
            self.model = PythonModel(my_model_func)
            self.answer = fake_single_point_data["Y"]

        def test_simple_mock_calibration(self):
            study = self.setup_study()
#            interface = study.subblocks["interface"]
#            rand_id = random.randint(0,10000000)
#            interface.add_line(InputFileLine("id_interface", f"'python_{rand_id}_id'"))
            study.set_core_limit(3)
            results = study.launch()
            self._simple_cal_assert_statements(results, self.tol)

        def test_simple_mock_calibration_do_not_save_eval_cache(self):
            study = self.setup_study()
#            interface = study.subblocks["interface"]
#            rand_id = random.randint(0,10000000)
#            interface.add_line(InputFileLine("id_interface",f"'python_{rand_id}_id'"))
            study.set_core_limit(3)
            study.do_not_save_evaluation_cache()
            results = study.launch()
            self._simple_cal_assert_statements(results, self.tol)

        def test_simple_mock_calibration_set_verbosity(self):
            study = self.setup_study()
#            interface = study.subblocks["interface"]
#            rand_id = random.randint(0,10000000)
#            interface.add_line(InputFileLine("id_interface", f"'python_{rand_id}_id'"))
            study.set_core_limit(3)
            study.set_output_verbosity("normal")
            results = study.launch()
            self._simple_cal_assert_statements(results, self.tol)

        def setup_study(self):
            study = self._mock_study(self.PC)
            study.add_evaluation_set(self.model, self._objectives, self.DC)
            self._set_study_specific_options(study)
            return study
    
        def _simple_cal_assert_statements(self, results, tol):
            self.assertAlmostEqual(results.outcome['best:a'], self.answer, delta=tol * self.answer)

        def _set_study_specific_options(self, study):
            pass
          

class TestDakotaColinyCobyla(DakotaMockStudyTestBase.CommonTests):
    base_file = os.path.join(get_current_files_path(__file__), "dakota_opt_ref_files", "coliny_cobyla_dakota.ref.base")
    target_file = "coliny_cobyla_dakota.ref"
    _mock_study = CobylaCalibrationStudy

    def setUp(self):
        super().setUp(__file__)

    def _set_study_specific_options(self, study):
        study.set_variable_tolerance(1e-7)

    def test_set_variable_tolerance(self):
        study = self.setup_study()
        study.set_variable_tolerance(1e-3)

class TestDakotaColinyPatternSearch(DakotaMockStudyTestBase.CommonTests):
    base_file = os.path.join(get_current_files_path(__file__), "dakota_opt_ref_files", "pattern_dakota.ref.base")
    target_file = "pattern_dakota.ref"
    _mock_study = PatternSearchCalibrationStudy

    def setUp(self):
        super().setUp(__file__)

    def test_input_options(self):
        study = self.setup_study()

        study.set_exploratory_moves("adaptive_pattern")
        study.set_solution_target(1.0)
        study.set_variable_tolerance(1e-2)


class TestDakotaColinyPatternSearchExploratoryMoves(DakotaMockStudyTestBase.CommonTests):
    _mock_study = PatternSearchCalibrationStudy

    def setUp(self):
        super().setUp(__file__)     
        self.tol=1e-3

    def _set_study_specific_options(self, study):
        study.set_exploratory_moves("adaptive_pattern")
        study.set_variable_tolerance(1e-2)


class TestDakotaColinySolisWets(DakotaMockStudyTestBase.CommonTests):
    _mock_study = SolisWetsCalibrationStudy

    def setUp(self):
        super().setUp(__file__)
        self.tol = 1e-3

    def test_input_options(self):
        study = self.setup_study()
        study.set_variable_tolerance(1e-3)

    def _set_study_specific_options(self, study):
        study.set_convergence_tolerance(5e-4)
        study.set_max_function_evaluations(100)


class TestDakotaGradient(DakotaMockStudyTestBase.CommonTests):
    _mock_study = GradientCalibrationStudy

    def setUp(self):
        super().setUp(__file__)

    def _set_study_specific_options(self, study):
        study.set_convergence_tolerance(5e-7)


class TestDakotaMeshAdaptiveSearch(DakotaMockStudyTestBase.CommonTests):
    _mock_study = MeshAdaptiveSearchCalibrationStudy

    def setUp(self):
        super().setUp(__file__)
        self.tol = 1e-4

    def test_input_options(self):
        study = self.setup_study()
        study.set_variable_neighborhood_search(0.25)
        study.set_variable_tolerance(1e-3)
    
    def _set_study_specific_options(self, study):
        study.set_variable_tolerance(1e-7)


class TestDakotaMeshAdaptiveSearchWithVariableNeighborhoodSearch(DakotaMockStudyTestBase.CommonTests):
    _mock_study = MeshAdaptiveSearchCalibrationStudy

    def setUp(self):
        super().setUp(__file__)

    def _set_study_specific_options(self, study):
        study.set_variable_neighborhood_search(0.5)


class TestDakotaMoga(DakotaMockStudyTestBase.CommonTests):
    _mock_study = MultiObjectiveGACalibrationStudy

    def setUp(self):
        super().setUp(__file__)

    def _simple_cal_assert_statements(self, results, tol):
        pass

    def _set_study_specific_options(self, study):
        study.set_max_iterations(10)
        study.set_population_size(20)
        study.set_core_limit(20)

    def setup_study(self):
        study = self._mock_study(self.PC)
        study.add_evaluation_set(self.model, self._objectives, self.DC)
        model2 = PythonModel(my_model_func_2)
        study.add_evaluation_set(model2, self._objectives, self.DC)
        self._set_study_specific_options(study)
        return study


class TestDakotaOptppPDS(DakotaMockStudyTestBase.CommonTests):
    _mock_study = ParallelDirectSearchCalibrationStudy

    def setUp(self):
        super().setUp(__file__)
        self.tol = 1e-3

    def test_input_options(self):
        study = self.setup_study()
        study.set_search_scheme_size(8)

    def _set_study_specific_options(self, study):
        study.set_convergence_tolerance(1e-5)


class TestDakotaOptppPDSSearchSchemeSize(DakotaMockStudyTestBase.CommonTests):
    _mock_study = ParallelDirectSearchCalibrationStudy

    def setUp(self):
        super().setUp(__file__)
        self.tol = 1e-3

    def _set_study_specific_options(self, study):
        study.set_search_scheme_size(8)
        study.set_convergence_tolerance(1e-5)


class TestDakotaSoga(DakotaMockStudyTestBase.CommonTests):
    _mock_study = SingleObjectiveGACalibrationStudy

    def setUp(self):
        super().setUp(__file__)

    def _simple_cal_assert_statements(self, results, tol):
        pass

    def _set_study_specific_options(self, study):
        study.set_max_iterations(1)
        study.set_population_size(20)
        study.set_core_limit(20)


