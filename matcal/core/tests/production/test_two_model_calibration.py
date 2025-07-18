## want this to kind of look like an input a user would write
import numpy as np

from matcal import *
from matcal.core.tests.MatcalUnitTest import MatcalUnitTest
from matcal.core.calibration_studies import (ScipyMinimizeStudy, 
                                            ScipyLeastSquaresStudy)


def python_model_1(param_1, param_2, state_param_1, state_param_2):
    import numpy as np
    x = np.linspace(0, 1, 100)
    response = param_1*state_param_2+param_2*x*x+state_param_1
    return {'x':x, 'response':response}


def python_model_2(param_1, param_2, state_param_1, state_param_2):
    import numpy as np
    x = np.linspace(0, 1, 100)
    response = param_1*state_param_1*x*x+param_2*x+state_param_2
    return {'x':x, 'response':response}


def run_calibration(ip, method=None, study=ScipyMinimizeStudy):
        param_1 = Parameter("param_1", 0, 10, ip[0])  # In MPa
        param_2 = Parameter("param_2", 0, 10, ip[1])
        parameter_collection = ParameterCollection("one_parameter", param_1, param_2)

        state_11 = State('state_11', state_param_1=1, state_param_2=1)
        state_10 = State('state_10', state_param_1=-1, state_param_2=0)
        state_01 = State('state_01', state_param_1=0, state_param_2=1)
        state_22 = State('state_22', state_param_1=2, state_param_2=-2)
        state_21 = State('state_21', state_param_1=-2, state_param_2=1)
        goal_params = {"param_1":np.sqrt(87), "param_2":np.e}

        mod1_data_state_11 = convert_dictionary_to_data(python_model_1(**goal_params, **state_11.params))
        mod1_data_state_11.set_state(state_11)
        mod1_data_state_10 = convert_dictionary_to_data(python_model_1(**goal_params, **state_10.params))
        mod1_data_state_10.set_state(state_10)
        mod1_data_state_01 = convert_dictionary_to_data(python_model_1(**goal_params, **state_01.params))
        mod1_data_state_01.set_state(state_01)
        mod1_data_state_22 = convert_dictionary_to_data(python_model_1(**goal_params, **state_22.params))
        mod1_data_state_22.set_state(state_22)
        mod1_data_state_21 = convert_dictionary_to_data(python_model_1(**goal_params, **state_21.params))
        mod1_data_state_21.set_state(state_21)

        mod1_dc = DataCollection("mod1", mod1_data_state_11, mod1_data_state_10, mod1_data_state_01, mod1_data_state_22, mod1_data_state_21)
        
        mod2_data_state_11 = convert_dictionary_to_data(python_model_2(**goal_params, **state_11.params))
        mod2_data_state_11.set_state(state_11)
        mod2_data_state_10 = convert_dictionary_to_data(python_model_2(**goal_params, **state_10.params))
        mod2_data_state_10.set_state(state_10)
        mod2_data_state_01 = convert_dictionary_to_data(python_model_2(**goal_params, **state_01.params))
        mod2_data_state_01.set_state(state_01)
        mod2_data_state_22 = convert_dictionary_to_data(python_model_2(**goal_params, **state_22.params))
        mod2_data_state_22.set_state(state_22)
        mod2_data_state_21 = convert_dictionary_to_data(python_model_2(**goal_params, **state_21.params))
        mod2_data_state_21.set_state(state_21)

        mod2_dc = DataCollection("mod2", mod2_data_state_11, mod2_data_state_10, mod2_data_state_01, 
                                 mod2_data_state_22, mod2_data_state_21)

        model1 = PythonModel(python_model_1)
        model2 = PythonModel(python_model_2) 

        objective_1 = CurveBasedInterpolatedObjective("x", "response")
        objective_2 = Objective("x", "response")
        objective_2.set_qoi_extractors(MaxExtractor("response"))
        objective_3 = Objective("x", "response")
        objective_3.set_qoi_extractors(MaxExtractor("x"))
        
        objectives = ObjectiveCollection('objectives', objective_1, objective_2, objective_3)

        calibration = study(parameter_collection, method=method)
        calibration.set_use_threads(always_use_threads=True)
        calibration.add_evaluation_set(model1, objectives, mod1_dc)

        simple_objective = CurveBasedInterpolatedObjective("x", "response")
        calibration.add_evaluation_set(model2, simple_objective, mod2_dc)
        calibration.set_core_limit(40)

        results = calibration.launch()
        return results, goal_params


class ScipyMinimizeMultiModelObjectiveTests(MatcalUnitTest):

    def setUp(self):
        super().setUp(__file__)

    def test_default_two_models_multiple_states_datasets_and_objectives(self):
        path = self.get_current_files_path(__file__)
        err_tol = .001
        results, goal_params = run_calibration([5,5])
        self.assertAlmostEqual(results.outcome['best:param_1'], goal_params["param_1"], delta=goal_params["param_1"] * err_tol)
        self.assertAlmostEqual(results.outcome['best:param_2'], goal_params["param_2"], delta=goal_params["param_2"] * err_tol)

    def test_default_different_initial_guess(self):
        path = self.get_current_files_path(__file__)
        err_tol = .001
        results, goal_params = run_calibration([3,7])
        self.assertAlmostEqual(results.outcome['best:param_1'], goal_params["param_1"], delta=goal_params["param_1"] * err_tol)
        self.assertAlmostEqual(results.outcome['best:param_2'], goal_params["param_2"], delta=goal_params["param_2"] * err_tol)

    def test_nelder_mead(self):
        path = self.get_current_files_path(__file__)
        err_tol = .001
        results, goal_params = run_calibration([3,7], method='nelder-mead')
        self.assertAlmostEqual(results.outcome['best:param_1'], goal_params["param_1"], delta=goal_params["param_1"] * err_tol)
        self.assertAlmostEqual(results.outcome['best:param_2'], goal_params["param_2"], delta=goal_params["param_2"] * err_tol)

    def test_dogleg(self):
        path = self.get_current_files_path(__file__)
        err_tol = .001
        results, goal_params = run_calibration([3,7], method='dogleg')
        self.assertAlmostEqual(results.outcome['best:param_1'], goal_params["param_1"], delta=goal_params["param_1"] * err_tol)
        self.assertAlmostEqual(results.outcome['best:param_2'], goal_params["param_2"], delta=goal_params["param_2"] * err_tol)

class ScipyLeastSquaresMultiModelObjectiveTests(MatcalUnitTest):

    def setUp(self):
        super().setUp(__file__)

    def test_two_models_multiple_states_datasets_and_objectives(self):
        path = self.get_current_files_path(__file__)
        err_tol = .001
        results, goal_params = run_calibration([5,5], study=ScipyLeastSquaresStudy)
        self.assertAlmostEqual(results.outcome['best:param_1'], goal_params["param_1"], delta=goal_params["param_1"] * err_tol)
        self.assertAlmostEqual(results.outcome['best:param_2'], goal_params["param_2"], delta=goal_params["param_2"] * err_tol)

    def test_different_initial_guess(self):
        path = self.get_current_files_path(__file__)
        err_tol = .001
        results, goal_params = run_calibration([3,7], study=ScipyLeastSquaresStudy)
        self.assertAlmostEqual(results.outcome['best:param_1'], goal_params["param_1"], delta=goal_params["param_1"] * err_tol)
        self.assertAlmostEqual(results.outcome['best:param_2'], goal_params["param_2"], delta=goal_params["param_2"] * err_tol)

    def test_lm(self):
        path = self.get_current_files_path(__file__)
        err_tol = .001
        results, goal_params = run_calibration([3,7], method='lm', study=ScipyLeastSquaresStudy)
        self.assertAlmostEqual(results.outcome['best:param_1'], goal_params["param_1"], delta=goal_params["param_1"] * err_tol)
        self.assertAlmostEqual(results.outcome['best:param_2'], goal_params["param_2"], delta=goal_params["param_2"] * err_tol)

    def test_dogbox(self):
        path = self.get_current_files_path(__file__)
        err_tol = .001
        results, goal_params = run_calibration([3,7], method='dogbox', study=ScipyLeastSquaresStudy)
        self.assertAlmostEqual(results.outcome['best:param_1'], goal_params["param_1"], delta=goal_params["param_1"] * err_tol)
        self.assertAlmostEqual(results.outcome['best:param_2'], goal_params["param_2"], delta=goal_params["param_2"] * err_tol)
