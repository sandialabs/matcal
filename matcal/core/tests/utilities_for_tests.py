import numpy as np
import os

from matcal.core.data import DataCollection, convert_dictionary_to_data
from matcal.core.models import ModelBase
from matcal.core.objective_results import ObjectiveQOI, ObjectiveResults
from matcal.core.simulators import Simulator, SimulatorResults
from matcal.core.state import State
from matcal.core.study_base import (StudyResults)
from matcal.core.utilities import (matcal_name_format, check_valid_matcal_name_string, 
  MatCalTypeStringError, MatcalNameFormatError, set_significant_figures, 
  make_clean_dir, get_current_time_string, check_item_is_correct_type, 
  check_value_is_positive, check_value_is_positive_integer, 
  check_value_is_positive_real, check_value_is_real_between_values, 
  check_value_is_nonempty_str)
from matcal.core.tests.MatcalUnitTest import MatcalUnitTest



class MockSimulator(Simulator):
    """
    Not intended for users: Mock simulator for tests.
    """

    def __init__(self, model_name, compute_information, results_information, state, template_dir='.'):
            super().__init__(model_name, compute_information, results_information, state,
                    template_dir='.')
            self._results = self.get_results()

    def run(self, parameters, working_dir=None):
        return SimulatorResults(self._results, "", "", None)

    def get_results(self, working_dir=None):
        results = self._results_information.read(self._results_information.results_filename)
        results.set_state(self._state)
        return results
    

class MockExecutableModel(ModelBase):
    """
    Not intended for users: Mock model for testing.
    """

    model_type = "mock_model"
    _simulator_class = MockSimulator

    def __init__(self, results_filename, executable='exec'):
        super().__init__(executable=executable)
        if not os.path.exists(results_filename):
            raise FileNotFoundError("Mock Model could not be created because "+
                                    "the mock results file could not be found.")
        self.set_results_filename(results_filename, file_type="csv")

    def set_results_filename(self, filename, file_type=None):
        super().set_results_filename(os.path.abspath(filename), file_type)

    def _get_simulator_class_inputs(self, state):
        args = [self.name, self._simulation_information, 
                self._results_information, state]
        kwargs = {}

        return args, kwargs
    @property
    def _input_file(self):
        my_filename = "mock_input.i"
        with open(my_filename, "w") as f:
            f.write("\n")
        return os.path.abspath(my_filename) 

    def _setup_state(self, state, preprocessor_arguments, template_dir=None):
        pass

    def _prepare_preprocessor_arguments(self, state, state_template_dir):
        pass


def _generate_singe_model_single_state_mock_eval_hist_given_params(params, 
                                                                   qoi_function, 
                                                      record_weighted_conditioned=False, 
                                                      best=None, 
                                                      results_save_frequency=1):
        state_name = "MockState"
        state = State(state_name, a=0)
        best.set_state(state)
        eval_key_order, eval_params = _process_params_for_eval_hist(params)
        
        all_qois = {}
        raw_objs = {}
        tot_obj = {}
        for eval_name, params in eval_params.items():
            fun_results, obj_qois, obj_obj, fun_dc = _generate_test_evaluations(qoi_function, 
                                                                                state, params)
            _append_evaluation(all_qois, raw_objs, tot_obj, eval_name, fun_results, obj_qois, 
                               obj_obj, fun_dc, best)
        
        results = StudyResults(record_weighted_conditioned=record_weighted_conditioned, 
                               results_save_frequency=results_save_frequency)
        results._initialize_evaluation_sets(all_qois, eval_key_order)
        results._update_parameter_history(eval_params, eval_key_order)   
        results._update_results_history(raw_objs, tot_obj, all_qois, eval_key_order)
        return results


def _generate_singe_model_single_state_mock_eval_hist(param_names, param_means, 
                                                      param_stds, n_samples, 
                                                      qoi_function, 
                                                      record_weighted_conditioned=False,
                                                      results_save_frequency=1,
                                                      best=None):
        state_name = "MockState"
        state = State(state_name, a=0)
        eval_key_order, eval_params = _generate_test_params(param_names, param_means, 
                                                            param_stds, n_samples)
        
        all_qois = {}
        raw_objs = {}
        tot_obj = {}
        for eval_name, params in eval_params.items():
            fun_results, obj_qois, obj_obj, fun_dc = _generate_test_evaluations(qoi_function, 
                                                                                state, params)
            _append_evaluation(all_qois, raw_objs, tot_obj, eval_name, fun_results, obj_qois, 
                               obj_obj, fun_dc, best)
        
        results = StudyResults(record_weighted_conditioned=record_weighted_conditioned,
                               results_save_frequency=results_save_frequency)
        results._initialize_evaluation_sets(all_qois, eval_key_order)
        results._update_parameter_history(eval_params, eval_key_order)   
        results._update_results_history(raw_objs, tot_obj, all_qois, eval_key_order)
        return results

def _append_evaluation(all_qois, raw_objs, tot_obj, eval_name, 
                       fun_results, obj_qois, obj_obj, fun_dc, best):
    obj_qois.set_simulation_qois(fun_dc)
    obj_qois.set_simulation_data(fun_dc)
    obj_qois.add_weighted_conditioned_simulation_qois(fun_results)
    if best is not None:
        
        best_dc = DataCollection("best", best)
        obj_qois.add_weighted_conditioned_experiment_qois(best)
        obj_qois.set_experiment_data(best_dc)
        obj_qois.set_experiment_qois(best_dc)
        resid = {}
        for field in best.keys():
            resid[field] = fun_results[field]-best[field]
        resid = convert_dictionary_to_data(resid)
        resid.set_state(fun_results.state)
        obj = {}
        tot_obj[eval_name] = 0
        for field in resid.keys():
            obj[field] = np.linalg.norm(resid[field])
            tot_obj[eval_name] += obj[field]
    else:
        resid = fun_results
        obj = {}
        for key in fun_results.keys():
            obj[key] = 1
        tot_obj[eval_name] = 1
        obj_qois.add_weighted_conditioned_experiment_qois(fun_results)
        obj_qois.set_experiment_data(fun_dc)
        obj_qois.set_experiment_qois(fun_dc)

    obj_obj.add_residuals(resid)
    obj_obj.add_weighted_conditioned_residuals(resid)
    obj = convert_dictionary_to_data(obj)
    obj.set_state(fun_results.state)        
    obj_obj.add_weighted_conditioned_objective(obj)

    all_qois[eval_name] = {"MockModel":{"MockObj":obj_qois}}
    raw_objs[eval_name] = {"MockModel":{"MockObj":obj_obj}}


def _generate_test_evaluations(qoi_function, state, params):
    fun_results = qoi_function(**params)
    fun_keys = list(fun_results.keys())
    obj_qois = ObjectiveQOI(fun_keys, fun_keys)
    raw_objs = ObjectiveResults(fun_keys, fun_keys)
    fun_results = convert_dictionary_to_data(fun_results)
    fun_results.set_state(state)
    fun_dc = DataCollection('fun', [fun_results])
    return fun_results,obj_qois,raw_objs,fun_dc


def _generate_test_params(names, means, stds, n_samples):
    params = _generate_random_params(names, means, stds, n_samples)
    return _process_params_for_eval_hist(params)


def _process_params_for_eval_hist(params):
    eval_key_order = []
    eval_params = {}
    first_key = list(params.keys())[0]
    for eval_i in range(len(params[first_key])):
        eval_name = f"eval.{eval_i+1}"
        eval_key_order.append(eval_name)
        eval_params[eval_name] = {}
        for p_name, p_val in params.items():
            eval_params[eval_name][p_name] = p_val[eval_i]
    return eval_key_order,eval_params

def _generate_random_params(names, means, stds, n_samples):
    params = {}
    for name, mean, std in zip(names, means, stds):
        params[name] = np.random.normal(mean, std, n_samples)
    return params



class TestCheckValidMatCalNameString(MatcalUnitTest):

  def setUp(self):
    super().setUp(__file__)

  def test_check_invalid_matcal_name_string(self):
    self.assert_error_type(MatCalTypeStringError, check_valid_matcal_name_string, "string_with_/")


  def test_check_valid_matcal_name_string(self):
    name = check_valid_matcal_name_string("valid")
    self.assertEqual(name, "valid")


class MatCalNameFormatTest(MatcalUnitTest):

  def setUp(self):
    super().setUp(__file__)
  
  def test_conversion(self):
  
    s = matcal_name_format("ALLCAPS")
    self.assertEqual(s,'ALLCAPS')
    
    s = matcal_name_format("spaces spaces")
    self.assertEqual(s,"spaces_spaces")
    s = matcal_name_format("Caps And SPACES")
    self.assertEqual(s,"Caps_And_SPACES")
    s = matcal_name_format("Mix_of Everything")
    self.assertEqual(s,"Mix_of_Everything")
    s = matcal_name_format("no_change_here")
    self.assertEqual(s,"no_change_here")
    
  def test_passList(self):
    list_of_strings = ["ALLCAPS","spaces Spaces","no_change"]
    result = matcal_name_format(list_of_strings)
    self.assertEqual(result[0],"ALLCAPS")
    self.assertEqual(result[1],"spaces_Spaces")
    self.assertEqual(result[2],"no_change")
    
  def test_errorCatch(self):
    self.assert_error_type(MatcalNameFormatError, matcal_name_format, 1)
    self.assert_error_type(MatcalNameFormatError, matcal_name_format,{})
    self.assert_error_type(MatcalNameFormatError, matcal_name_format,[])
    self.assert_error_type(MatcalNameFormatError, matcal_name_format,"")
    self.assert_error_type(MatcalNameFormatError, matcal_name_format,[1,2])
    self.assert_error_type(MatcalNameFormatError, matcal_name_format,None)

class TestSetSignificantFigures(MatcalUnitTest):
  def setUp(self):
    super().setUp(__file__)
    self.test_inputs = [
      1.114,  # positive, round down
      1.115,  # positive, round up
      -1.114,  # negative
      1.114e-16,  # extremely small
      1.114e16,  # extremely large
      0,  # zero
      2.112,
      float('inf'),  # infinite
    ]

    self.test_inputs_array_like = [[1.114, 1.115e-16], np.array([1.115, 1.114e-16])]
  def test_set_significant_figures_two(self):
    solutions = [
      1.1,  # positive, round down
      1.1,  # positive, round up
      -1.1,  # negative
      1.1e-16,  # extremely small
      1.1e16,  # extremely large
      0,  # zero
      2.1,
      float('inf'),  # infinite
      ]

    test_results = []
    for x in self.test_inputs:
       test_results.append(set_significant_figures(x, 2))

    for test_result, solution in zip(test_results,solutions):
      self.assertEqual(test_result, solution)

  def test_set_significant_figures_three(self):
    solutions = [
      1.11,  # positive, round down
      1.12,  # positive, round up
      -1.11,  # negative
      1.11e-16,  # extremely small
      1.11e16,  # extremely large
      0,  # zeros
      2.11,
      float('inf'),  # infinite
      ]

    test_results = []
    for x in self.test_inputs:
       test_results.append(set_significant_figures(x, 3))

    for test_result, solution in zip(test_results,solutions):
      self.assertEqual(test_result, solution)

  def test_array_like_set_significant_figures_three(self):
    solutions =  [[1.11, 1.12e-16], np.array([1.12, 1.11e-16])]

    test_results = []
    for x in self.test_inputs_array_like:
       test_results.append(set_significant_figures(x, 3))

    for test_result, solution in zip(test_results,solutions):
      for result_entry, solution_entry in zip(test_result, solution):
        self.assertEqual(result_entry, solution_entry)

class TestBasicUtilities(MatcalUnitTest):
  def setUp(self):
    super().setUp(__file__)

  def test_make_clean_dir(self):
    import os
    os.mkdir("test")
    os.mkdir("test/test_nested")
    make_clean_dir("test")
    self.assertTrue(os.path.exists("test"))

  def test_get_current_time_string(self):
    cur_time_str = get_current_time_string()
    self.assertIsInstance(cur_time_str, str)
    self.assertEqual(len(cur_time_str.split(":")), 3)
    self.assertEqual(len(cur_time_str.split("-")), 3)

  def test_check_item_is_correct_type(self):
    with self.assertRaises(TypeError):
      check_item_is_correct_type(1, str, "test", "test")
  
  def test_check_value_is_positive(self):
    with self.assertRaises(ValueError):
      check_value_is_positive(-0.5, "test", "test")

  def test_check_value_is_positive_integer(self):
    with self.assertRaises(TypeError):
      check_value_is_positive_integer(-0.5, "test", "test")
    with self.assertRaises(ValueError):
      check_value_is_positive_integer(-5, "test", "test")

  def test_check_value_is_positive_real(self):
    with self.assertRaises(ValueError):
      check_value_is_positive_real(-0.5, "test", "test")

  def test_check_value_is_real_between_values(self):
    with self.assertRaises(ValueError):
      check_value_is_real_between_values(1, 1, 2, "test", "test")
    check_value_is_real_between_values(1, 1, 2, "test", "test", True)
    with self.assertRaises(ValueError):
      check_value_is_real_between_values(0.5, 1, 2, "test", "test", True)

  def test_check_value_is_nonempty_str(self):
    with self.assertRaises(TypeError):
      check_value_is_nonempty_str(1, "test", "test")
    with self.assertRaises(ValueError):
      check_value_is_nonempty_str("", "test", "test")