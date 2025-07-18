import os
from matcal.core.logger import matcal_print_message
import numpy as np
import shutil

from matcal.core import *
from matcal.core.data import convert_dictionary_to_data
from matcal.dakota.local_calibration_studies import GradientCalibrationStudy
from matcal.core.tests.MatcalUnitTest import MatcalUnitTest


def my_function(**param_dict):
    import numpy as np
    x = np.linspace(0, 10, 25)
    y = param_dict['m'] * x
    if "restart_kill" in param_dict:
        if param_dict['m'] < 1e6:
            return None
    return {"x": x, "y": y}

def get_goal_data():
    return convert_dictionary_to_data(my_function(m=2))

class OneParameterPythonTest(MatcalUnitTest):
    def setUp(self):
        super().setUp(__file__)
        self.error_tol = 1e-6

    def test_calibration(self):
        err_tol = .002

        slope = Parameter("m", 0., 100.)
        parameter_collection = ParameterCollection("one_parameter", slope)

        curve_data = get_goal_data()

        model = PythonModel(my_function)

        calibration = GradientCalibrationStudy(parameter_collection)
        objective = CurveBasedInterpolatedObjective("x", "y")
        calibration.add_evaluation_set(model, objective, curve_data)
        calibration.set_core_limit(6)

        results = calibration.launch()
        self.assertAlmostEqual(results.outcome["best:m"], 2., delta=2 * self.error_tol)

    def test_calibration_max_extractor_obj_multiple_data_sets(self):

        slope = Parameter("m", 0., 100.)
        parameter_collection = ParameterCollection("one_parameter", slope)

        curve_data = get_goal_data()
        curve_data_2 = curve_data.copy()

        model = PythonModel(my_function)

        calibration = GradientCalibrationStudy(parameter_collection)
        objective = Objective("x", "y")
        objective.set_qoi_extractors(MaxExtractor("y"))
        dc = DataCollection("test", curve_data, curve_data_2)
        calibration.add_evaluation_set(model, objective, dc)
        calibration.set_core_limit(6)

        results = calibration.launch()
        self.assertAlmostEqual(results.outcome["best:m"], 2., delta=2 * self.error_tol)

    def test_calibration_small_steps(self):
        step_size = 1e-3

        slope = Parameter("m", 0., 100.)
        parameter_collection = ParameterCollection("one_parameter", slope)

        curve_data = get_goal_data()

        model = PythonModel(my_function)

        calibration = GradientCalibrationStudy(parameter_collection)
        objective = CurveBasedInterpolatedObjective("x", "y")
        calibration.add_evaluation_set(model, objective, curve_data)
        calibration.set_core_limit(6)
        calibration.set_step_size(step_size)

        results = calibration.launch()
        self._confirm_step_size(step_size)
        self.assertAlmostEqual(results.outcome["best:m"], 2., delta=2 * self.error_tol)

    def _confirm_step_size(self, step_size):
        dakota_file = 'dakota.in'
        have_found = False
        with open(dakota_file, 'r') as df:
            for line in df:
                if self._is_step_size_line(line):
                    self.assertAlmostEqual(np.log10(self._parse_step_size(line)), np.log10(step_size))
                    have_found = True
        self.assertTrue(have_found)

    def _is_step_size_line(self, line):
        ss_key = "fd_step_size"
        s_line = self._clean_line(line)
        if len(s_line) > 0:
            return s_line[0] == ss_key
        else:
            return False

    def _clean_line(self, line):
        s_line = line.strip()
        s_line = s_line.split()
        return s_line

    def _parse_step_size(self, line):
        s_line = self._clean_line(line)
        return float(s_line[-1])

    def test_calibration_deactivate_cache(self):
        slope = Parameter("m", 0., 100.)
        parameter_collection = ParameterCollection("one_parameter", slope)

        curve_data = get_goal_data()
        model = PythonModel(my_function)

        calibration = GradientCalibrationStudy(parameter_collection)
        calibration.do_not_save_evaluation_cache()
        objective = CurveBasedInterpolatedObjective("x", "y")
        calibration.add_evaluation_set(model, objective, curve_data)
        calibration.set_core_limit(6)

        results = calibration.launch()
        self.assertAlmostEqual(results.outcome["best:m"], 2., delta=2 * self.error_tol)
   
    def test_calibration_with_restart_write_restart(self):
        slope = Parameter("m", 0., 1e9, 5.2e8)
        parameter_collection = ParameterCollection("one_parameter", slope)
        curve_data = get_goal_data()
        model = PythonModel(my_function)
        model.add_constants(restart_kill="True")
        calibration = GradientCalibrationStudy(parameter_collection)
        restart_filename = "my_restart_test.rst"
        
        calibration.set_restart_filename(restart_filename)

        objective = CurveBasedInterpolatedObjective("x", "y")
        calibration.add_evaluation_set(model, objective, curve_data)
        calibration.set_core_limit(6)
        with self.assertRaises(AttributeError):
            #catches desired failure once calibration gets close to solution
            results = calibration.launch()
        old_results = matcal_load("in_progress_results.joblib")    
        old_results_end_obj = old_results.best_total_objective
        restart_idx = len(old_results.total_objective_history)
        model.reset_constants()
        slope = Parameter("m", 0., 1e9, 5.2e8)
        parameter_collection = ParameterCollection("one_parameter", slope)
        calibration = GradientCalibrationStudy(parameter_collection)
        interface_block = calibration.get_interface_block()
        id_line = interface_block.lines["id_interface"]
        id_line_val = interface_block.get_line_value("id_interface")
        id_line_val = int(id_line_val.split("_")[1])
        id_line.set(f"'python_{int(id_line_val)-1}_id'")

        calibration.add_evaluation_set(model, objective, curve_data)
        calibration.set_core_limit(6)
        calibration.restart(restart_filename)
        results = calibration.launch()
        
        with open("dakota.out", "r") as f:
            output = f.read()
        #if the original first value of the objective, is equal to 
        #the value at the current restart, then the restarted calibration
        # is essentially starting over. So something is wrong with restarts
        # even though dakota is restarting. 
        self.assertNotEqual(old_results.total_objective_history[0], 
                               results.total_objective_history[restart_idx])
        self.assertAlmostEqual(results.outcome["best:m"], 2., delta=2 * self.error_tol)
        self.assertTrue(f"Reading restart file \'{os.path.abspath(restart_filename)}\'." in output)
        self.assertTrue(f"Restart record    1" in output)

    def test_calibration_with_restart_write_restart_subworkdir(self):
        slope = Parameter("m", 0., 1e9, 5.4e8)
        parameter_collection = ParameterCollection("one_parameter", slope)
        curve_data = get_goal_data()
        model = PythonModel(my_function)
        model.add_constants(restart_kill="True")
        calibration = GradientCalibrationStudy(parameter_collection)
        restart_filename = "my_restart_test.rst"
       
        calibration.set_restart_filename(restart_filename)
        calibration.set_working_directory("test")
        objective = CurveBasedInterpolatedObjective("x", "y")
        calibration.add_evaluation_set(model, objective, curve_data)
        calibration.set_core_limit(6)
        init_dir = os.getcwd()
        with self.assertRaises(AttributeError):
            #catches desired failure once calibration gets close to solution
            results = calibration.launch()
        os.chdir(init_dir)
        model.reset_constants()
        slope = Parameter("m", 0., 1e9, 5.5e8)
        parameter_collection = ParameterCollection("one_parameter", slope)
        calibration = GradientCalibrationStudy(parameter_collection)
        calibration.add_evaluation_set(model, objective, curve_data)
        calibration.set_working_directory("test")

        calibration.set_core_limit(6)
        restarted_restart_file = os.path.join("test",restart_filename)
        restarted_results_file = os.path.join("test","in_progress_results.joblib")

        calibration.restart(restarted_restart_file, restarted_results_file)
        goal_restart = os.path.abspath(restarted_restart_file)
        results = calibration.launch()
        
        with open(os.path.join("test", "dakota.out"), "r") as f:
            output = f.read()

        self.assertAlmostEqual(results.outcome["best:m"], 2., delta=2 * self.error_tol)
        self.assertTrue(f"Reading restart file \'{goal_restart}\'." in output)
        self.assertTrue(f"Restart record    1" in output)

    def test_user_weighted_calibration(self):
        slope = Parameter("m", 0., 100., 5)
        parameter_collection = ParameterCollection("one_parameter", slope)
        curve_data = convert_dictionary_to_data({"x":[0,1,2,3,4,5,6,7,8], "y":[0,2,4,6,8,5,6,7,8]})
        model = PythonModel(my_function)

        import numpy as np
        def wfun(x_field, target_field, residual):
            import numpy as np
            w = x_field > 5
            return np.multiply(w, residual)
        weighting = UserFunctionWeighting('x', 'y', wfun)
        objective = CurveBasedInterpolatedObjective("x", "y")
        objective.set_metric_function(L1NormMetricFunction())
        objective.set_field_weights( weighting)

        calibration = GradientCalibrationStudy(parameter_collection)
        calibration.add_evaluation_set(model, objective, curve_data)
        calibration.set_core_limit(6)

        results = calibration.launch()
        self.assertAlmostEqual(results.outcome["best:m"], 1., delta=1 * self.error_tol)

def _make_random_seed():
    import time
    seed = int(str(time.time()).replace('.', '')[-9:-1])
    return seed

def my_cubic_function(**p):
    import numpy as np
    import time
    x = np.linspace(0, 10, 25)
    y = p['c0']
    y += np.power(p['c1']*x, 1)
    y += np.power(p['c2']*x, 2)
    y += np.power(p['c3']*x, 3)
    def _make_random_seed():
        import time
        seed = int(str(time.time()).replace('.', '')[-9:-1])
        return seed
    seed = _make_random_seed()
    np.random.seed(seed)
    sleep_time = np.random.uniform(0,.5)
    matcal_print_message(f"sleeping: {sleep_time} for {p}")
    time.sleep(sleep_time)
    return {"x": x, "y": y}


class DisorganizedJobReturnPythonTest(MatcalUnitTest):

    def setUp(self):
        super().setUp(__file__)
        self._delta = 1e-6
    
    def test_random_return_full_parallel(self):
        model, objective, calibration = self._cubic_setup()

        goal = {'c0':1.5, 'c1':0, 'c2':8, 'c3': 6.25}
        goal_dict = my_cubic_function(**goal)
        curve_data = convert_dictionary_to_data(goal_dict)

        calibration.add_evaluation_set(model, objective, curve_data)
        calibration.set_core_limit(6)

        results = calibration.launch()
        for name, value in goal.items():
            self.assertAlmostEqual(value, results.outcome[f'best:{name}'], delta=self._delta)

    def test_random_return_subset_parallel(self):
        model, objective, calibration = self._cubic_setup()

        goal = {'c0':1.5, 'c1':3.14159, 'c2':8, 'c3': 6.25}
        goal_dict = my_cubic_function(**goal)
        curve_data = convert_dictionary_to_data(goal_dict)

        calibration.add_evaluation_set(model, objective, curve_data)
        calibration.set_core_limit(2)

        results = calibration.launch()
        for name, value in goal.items():
            self.assertAlmostEqual(value, results.outcome[f'best:{name}'], delta=self._delta)

    def test_random_return_subset_weight_parallel(self):
        #NOTE This test failed in a seemingly random manner on 11-7-2022, for git 
        # commit 58b6e59c8628a2c7d237218573bb1d5414cae22c. Not sure why, but it 
        # passed when ran again. Maybe related to the seed setting?
        # Initially failed when ran with all other production tests, 
        # then passed individually, then passed when run as group no code changes between.
        model, objective, calibration = self._cubic_setup()

        goal = {'c0':1.5, 'c1':3.14159, 'c2':8, 'c3': 6.25}
        goal_dict = my_cubic_function(**goal)
        curve_data = convert_dictionary_to_data(goal_dict)

        def my_weight_function(x_field, target_field, residual):
            import numpy as np
            w = x_field > 5
            return np.multiply(w, residual)

        special_weighting = UserFunctionWeighting('x', 'y', my_weight_function)
        objective.set_field_weights(special_weighting)

        calibration.add_evaluation_set(model, objective, curve_data)
        calibration.set_core_limit(2)

        results = calibration.launch()
        for name, value in goal.items():
            self.assertAlmostEqual(value, results.outcome[f'best:{name}'], delta=self._delta)


    def _cubic_setup(self):
        import numpy as np
        np.random.seed(_make_random_seed())
        c0 = Parameter('c0', 0, 10, np.random.uniform(1, 3))
        c1 = Parameter('c1', 0, 10, np.random.uniform(2, 4))
        c2 = Parameter('c2', 0, 10, np.random.uniform(7, 9))
        c3 = Parameter('c3', 0, 10, np.random.uniform(5, 7))

        calibration = GradientCalibrationStudy(c0, c1, c2, c3)


        model = PythonModel(my_cubic_function)
        objective = CurveBasedInterpolatedObjective('x', 'y')
        return model,objective, calibration