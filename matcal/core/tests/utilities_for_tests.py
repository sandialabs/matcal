import numpy as np
import os

from matcal.core.data import DataCollection, convert_dictionary_to_data
from matcal.core.models import ModelBase
from matcal.core.objective_results import ObjectiveQOI, ObjectiveResults
from matcal.core.simulators import Simulator, SimulatorResults
from matcal.core.state import State
from matcal.core.study_base import (StudyResults)


class MockSimulator(Simulator):
    """
    Not intended for users: Mock simulator for tests.
    """

    def __init__(self, model_name, compute_information, results_information, 
                 state, template_dir='.'):
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



