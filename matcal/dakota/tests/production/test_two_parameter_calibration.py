import numpy as np

from matcal.core import *
from matcal.core.tests.MatcalUnitTest import MatcalUnitTest

from matcal.dakota.local_calibration_studies import GradientCalibrationStudy


def my_function(**param_dict):
    import numpy as np
    def helper_function(b, y):
        return b + y
    x = np.linspace(0, 10, 25)
    y = param_dict['m'] * x
    y = helper_function(param_dict["b"], y)
    return {"x": x, "y": y}

def get_goal_data():
    return convert_dictionary_to_data(my_function(m=2, b=0))


def my_squared_function(**param_dict):
    import numpy as np
    def helper_function(b, y):
        return b + y
    x = np.linspace(0, 10, 25)
    y = -param_dict['m'] * (x-1)**2
    y = helper_function(param_dict["b"], y)
    return {"x": x, "y": y}


class TwoParameterPythonTest(MatcalUnitTest):
    def setUp(self):
        super().setUp(__file__)
        self.error_tol = 1e-6


    def evaluate_calibration_results_for_method(self, method="nl2sol", 
                                                convergence_tol=1e-12, 
                                                max_iterations=25, 
                                                step_size = 1e-7):
        slope = Parameter("m", 0., 100.)
        intercept = Parameter("b", -5, 5, 0.5+1e-2*np.random.uniform())
        parameter_collection = ParameterCollection("one_parameter", slope, intercept)

        curve_data = get_goal_data()
        model = PythonModel(my_function)

        calibration = GradientCalibrationStudy(parameter_collection)
        calibration.set_method(method)

        objective = CurveBasedInterpolatedObjective("x", "y")
        calibration.add_evaluation_set(model, objective, curve_data)
        calibration.set_core_limit(6)
        calibration.set_max_iterations(max_iterations)
        calibration.set_convergence_tolerance(convergence_tol)
        calibration.set_step_size(step_size)
        results = calibration.launch()

        self.assertAlmostEqual(results.outcome["best:m"], 2., delta=2 * self.error_tol)
        self.assertAlmostEqual(results.outcome["best:b"], 0, delta=self.error_tol)


    def test_calibration_nl2sol(self):
        self.evaluate_calibration_results_for_method("nl2sol")

    def test_calibration_optpp_g_newton(self):
        self.error_tol = 1e-3
        self.evaluate_calibration_results_for_method("optpp_g_newton", 
                                                     max_iterations=25, 
                                                     convergence_tol=1e-10,
                                                     step_size=1e-6)
       
    def test_calibration_nlssol_sqp(self):
        self.evaluate_calibration_results_for_method("nlssol_sqp")

    def test_calibration_npsol_sqp(self):
        self.error_tol = 1e-4
        self.evaluate_calibration_results_for_method("npsol_sqp", convergence_tol=1e-6)
    
    def test_calibration_dot_sqp(self):
        #self.error_tol = 1e-2
        self.evaluate_calibration_results_for_method("dot_sqp", 
                                                     convergence_tol=1e-9, 
                                                     step_size=1e-6,
                                                     max_iterations=30)

    def test_calibration_dot_sqp_inf_norm_metric(self):
        self.error_tol = 1e-2
        slope = Parameter("m", 0., 100.)
        intercept = Parameter("b", -5, 5, 1)
        parameter_collection = ParameterCollection("one_parameter", slope, intercept)

        curve_data = get_goal_data()

        model = PythonModel(my_function)

        calibration = GradientCalibrationStudy(parameter_collection)
        calibration.set_method("dot_sqp")
        objective = CurveBasedInterpolatedObjective("x", "y")
        import numpy as np
        objective.set_metric_function(NormMetricFunction(np.inf))
        objective2 = CurveBasedInterpolatedObjective("x", "y")
        calibration.add_evaluation_set(model, objective, curve_data)
        calibration.add_evaluation_set(model, objective2, curve_data)
        
        calibration.set_convergence_tolerance(1e-5)
        calibration.set_step_size(1e-5)
        calibration.set_core_limit(6)
        calibration.set_output_verbosity("verbose")
        results = calibration.launch()

        self.assertAlmostEqual(results.outcome["best:m"], 2., delta=2 * self.error_tol)
        self.assertAlmostEqual(results.outcome["best:b"], 0, delta=self.error_tol)

    def test_calibration_conmin_mfd(self):
        self.error_tol= 1e-5
        self.evaluate_calibration_results_for_method("conmin_mfd")

    def test_calibration_optpp_q_newton(self):
        self.error_tol=1e-2
        self.evaluate_calibration_results_for_method("optpp_q_newton")

    def test_calibration_two_models_one_objective(self):
        slope = Parameter("m", 0., 100.)
        intercept = Parameter("b", -5, 5, 1)
        parameter_collection = ParameterCollection("one_parameter", slope, intercept)

        curve_data = get_goal_data()
        squared_values = my_squared_function(m = 2, b = 0)
        squared_data = convert_dictionary_to_data(squared_values)

        model = PythonModel(my_function)
        squared_model = PythonModel(my_squared_function)

        calibration = GradientCalibrationStudy(parameter_collection)
        curve_objective = CurveBasedInterpolatedObjective("x", "y")

        calibration.add_evaluation_set(model, curve_objective, curve_data)
        calibration.add_evaluation_set(squared_model, curve_objective, squared_data)

        calibration.set_core_limit(6)

        results = calibration.launch()

        self.assertAlmostEqual(results.outcome["best:m"], 2., delta=2 * self.error_tol)
        self.assertAlmostEqual(results.outcome["best:b"], 0, delta=self.error_tol)

    def test_calibration_two_models_two_objectives_through_add_eval_set(self):
        slope = Parameter("m", 0., 100.)
        intercept = Parameter("b", -5, 5, 1)
        parameter_collection = ParameterCollection("one_parameter", slope, intercept)

        curve_data = get_goal_data()
        squared_values = my_squared_function(m=2, b=0)
        squared_data = convert_dictionary_to_data(squared_values)

        model = PythonModel(my_function)
        squared_model = PythonModel(my_squared_function)

        calibration = GradientCalibrationStudy(parameter_collection)
        curve_objective = CurveBasedInterpolatedObjective("x", "y")
        max_objective = Objective("y")
        max_objective.set_qoi_extractors(MaxExtractor("y"))

        calibration.add_evaluation_set(model, curve_objective, curve_data)

        calibration.add_evaluation_set(squared_model, curve_objective, squared_data)
        calibration.add_evaluation_set(squared_model, max_objective, squared_data)


        calibration.set_core_limit(6)

        results = calibration.launch()

        self.assertAlmostEqual(results.outcome["best:m"], 2., delta=2 * self.error_tol)
        self.assertAlmostEqual(results.outcome["best:b"], 0, delta=self.error_tol)

    def test_calibration_two_models_two_objectives_through_objective_collection(self):
        slope = Parameter("m", 0., 100)
        intercept = Parameter("b", -5, 5, 1)
        parameter_collection = ParameterCollection("one_parameter", slope, intercept)

        curve_data = get_goal_data()
        squared_values = my_squared_function(m=2, b=0)
        squared_data = convert_dictionary_to_data(squared_values)

        model = PythonModel(my_function)
        squared_model = PythonModel(my_squared_function)

        calibration = GradientCalibrationStudy(parameter_collection)
        curve_objective = CurveBasedInterpolatedObjective("x", "y")
        max_objective = Objective("y")
        max_objective.set_qoi_extractors(MaxExtractor("y"))

        obj_c = ObjectiveCollection("my objs", curve_objective, max_objective)

        calibration.add_evaluation_set(model, curve_objective, curve_data)

        calibration.add_evaluation_set(squared_model, obj_c, squared_data)

        calibration.set_core_limit(6)

        results = calibration.launch()

        self.assertAlmostEqual(results.outcome["best:m"], 2., delta=2 * self.error_tol)
        self.assertAlmostEqual(results.outcome["best:b"], 0, delta=self.error_tol)


    def test_calibration_two_models_one_objective_use_threads(self):
        slope = Parameter("m", 0., 100.)
        intercept = Parameter("b", -5, 5, 1)
        parameter_collection = ParameterCollection("one_parameter", slope, intercept)

        curve_data = get_goal_data()
        squared_values = my_squared_function(m = 2, b = 0)
        squared_data = convert_dictionary_to_data(squared_values)

        model = PythonModel(my_function)
        squared_model = PythonModel(my_squared_function)

        calibration = GradientCalibrationStudy(parameter_collection)
        curve_objective = CurveBasedInterpolatedObjective("x", "y")

        calibration.add_evaluation_set(model, curve_objective, curve_data)
        calibration.add_evaluation_set(squared_model, curve_objective, squared_data)
        calibration.set_use_threads()
        calibration.set_core_limit(1)
        results = calibration.launch()

        self.assertAlmostEqual(results.outcome["best:m"], 2., delta=2 * self.error_tol)
        self.assertAlmostEqual(results.outcome["best:b"], 0, delta=self.error_tol)

    def test_calibration_two_models_one_objective_always_use_threads(self):
        slope = Parameter("m", 0., 100.)
        intercept = Parameter("b", -5, 5, 1)
        parameter_collection = ParameterCollection("one_parameter", slope, intercept)

        curve_data = get_goal_data()
        squared_values = my_squared_function(m = 2, b = 0)
        squared_data = convert_dictionary_to_data(squared_values)

        model = PythonModel(my_function)
        squared_model = PythonModel(my_squared_function)

        calibration = GradientCalibrationStudy(parameter_collection)
        curve_objective = CurveBasedInterpolatedObjective("x", "y")

        calibration.add_evaluation_set(model, curve_objective, curve_data)
        calibration.add_evaluation_set(squared_model, curve_objective, squared_data)
        calibration.set_use_threads(always_use_threads=True)
        calibration.set_core_limit(10)
        results = calibration.launch()

        self.assertAlmostEqual(results.outcome["best:m"], 2., delta=2 * self.error_tol)
        self.assertAlmostEqual(results.outcome["best:b"], 0, delta=self.error_tol)