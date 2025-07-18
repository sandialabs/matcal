import matcal as mc
from matcal.core.tests.MatcalUnitTest import MatcalUnitTest
import numpy as np
import time

def quad(a,b):
    import numpy as np
    n_pts = 30
    x = np.linspace(0, 1, n_pts)
    y = b*x + a * np.power(x, 2)
    import time
    time.sleep(.1)
    return x, y

def model_wrapper(**params):
    a = params['a']
    b = params['b']
    out = quad(a,b)
    return {'x':out[0], 'y':out[1]}

class TestParameterStudy(MatcalUnitTest):

    def setUp(self):
        super().setUp(__file__)

    def test_restart_parameter_study_gets_same_answers_serial(self):
        a = mc.Parameter("a", 0, 1)
        b = mc.Parameter("b", 0, 1)
        pc = mc.ParameterCollection('ab', a, b)

        model = mc.PythonModel(model_wrapper)

        x_ref = np.linspace(0, 1, 10)
        obj = mc.SimulationResultsSynchronizer('x', x_ref, 'y')

        n_evals = 20
        eval_dict = {'a':np.random.uniform(0, 1, n_evals), 'b':np.random.uniform(0, 1, n_evals)}

        study = mc.ParameterStudy(pc)
        study.run_in_serial()
        study.add_evaluation_set(model, obj)
        for i_eval in range(n_evals):
            a_val = eval_dict['a'][i_eval]
            b_val = eval_dict['b'][i_eval]
            study.add_parameter_evaluation(a=a_val, b=b_val)

        base_start = time.time()
        results = study.launch()
        base_end = time.time()
        base_time_delta = base_end - base_start

        study2 = mc.ParameterStudy(pc)
        study2.run_in_serial()
        study2.add_evaluation_set(model, obj)
        study2.restart()
        for i_eval in range(n_evals):
            a_val = eval_dict['a'][i_eval]
            b_val = eval_dict['b'][i_eval]
            study2.add_parameter_evaluation(a=a_val, b=b_val)

        restart_start = time.time()
        restart_results = study2.launch()
        restart_end = time.time()
        restart_time_delta = restart_end - restart_start

        expected_speedup = 4.
        speedup = base_time_delta / restart_time_delta
        self.assertGreaterEqual(speedup, expected_speedup)

        base_hist = results.simulation_history[model.name]["matcal_default_state"]
        restart_hist = restart_results.simulation_history[model.name]["matcal_default_state"]
        for base, restart in zip(base_hist, restart_hist):
            self.assert_close_dicts_or_data(base, restart)

    def test_restart_parameter_study_gets_same_answers_distributed(self):
        a = mc.Parameter("a", 0, 1)
        b = mc.Parameter("b", 0, 1)
        pc = mc.ParameterCollection('ab', a, b)

        model = mc.PythonModel(model_wrapper)

        x_ref = np.linspace(0, 1, 10)
        obj = mc.SimulationResultsSynchronizer('x', x_ref, 'y')


        n_evals = 50
        eval_dict = {'a':np.random.uniform(0, 1, n_evals),
                      'b':np.random.uniform(0, 1, n_evals)}

        study = mc.ParameterStudy(pc)
        study.add_evaluation_set(model, obj)
        for i_eval in range(n_evals):
            a_val = eval_dict['a'][i_eval]
            b_val = eval_dict['b'][i_eval]
            study.add_parameter_evaluation(a=a_val, b=b_val)

        base_start = time.time()
        results = study.launch()
        base_end = time.time()
        base_time_delta = base_end - base_start

        study2 = mc.ParameterStudy(pc)
        study2.add_evaluation_set(model, obj)
        study2.restart()
        for i_eval in range(n_evals):
            a_val = eval_dict['a'][i_eval]
            b_val = eval_dict['b'][i_eval]
            study2.add_parameter_evaluation(a=a_val, b=b_val)

        restart_start = time.time()
        restart_results = study2.launch()
        restart_end = time.time()
        restart_time_delta = restart_end - restart_start

        expected_speedup = 4.
        speedup = base_time_delta / restart_time_delta

        self.assertGreaterEqual(speedup, expected_speedup)

        base_hist = results.simulation_history[model.name]["matcal_default_state"]
        restart_hist = restart_results.simulation_history[model.name]["matcal_default_state"]
        for base, restart in zip(base_hist, restart_hist):
            self.assert_close_dicts_or_data(base, restart)

    def test_restart_parameter_study_gets_same_answers_distributed_parallel(self):
        a = mc.Parameter("a", 0, 1)
        b = mc.Parameter("b", 0, 1)
        pc = mc.ParameterCollection('ab', a, b)

        model = mc.PythonModel(model_wrapper)

        x_ref = np.linspace(0, 1, 10)
        obj = mc.SimulationResultsSynchronizer('x', x_ref, 'y')

        n_evals = 50
        eval_dict = {'a':np.random.uniform(0, 1, n_evals), 
                     'b':np.random.uniform(0, 1, n_evals)}

        study = mc.ParameterStudy(pc)
        study.add_evaluation_set(model, obj)
        n_cores = 4
        study.set_core_limit(n_cores)
        for i_eval in range(n_evals):
            a_val = eval_dict['a'][i_eval]
            b_val = eval_dict['b'][i_eval]
            study.add_parameter_evaluation(a=a_val, b=b_val)

        base_start = time.time()
        results = study.launch()
        base_end = time.time()
        base_time_delta = base_end - base_start

        study2 = mc.ParameterStudy(pc)
        study2.add_evaluation_set(model, obj)
        study2.set_core_limit(n_cores)
        study2.restart()
        for i_eval in range(n_evals):
            a_val = eval_dict['a'][i_eval]
            b_val = eval_dict['b'][i_eval]
            study2.add_parameter_evaluation(a=a_val, b=b_val)

        restart_start = time.time()
        restart_results = study2.launch()
        restart_end = time.time()
        restart_time_delta = restart_end - restart_start

        expected_speedup = 2.
        speedup = base_time_delta / restart_time_delta

        self.assertGreaterEqual(speedup, expected_speedup)

        base_hist = results.simulation_history[model.name]["matcal_default_state"]
        restart_hist = restart_results.simulation_history[model.name]["matcal_default_state"]
        for base, restart in zip(base_hist, restart_hist):
            self.assert_close_dicts_or_data(base, restart)

    def test_restart_parameter_study_raise_error_for_parallel_threads(self):
        a = mc.Parameter("a", 0, 1)
        b = mc.Parameter("b", 0, 1)
        pc = mc.ParameterCollection('ab', a, b)

        model = mc.PythonModel(model_wrapper)

        x_ref = np.linspace(0, 1, 10)
        obj = mc.SimulationResultsSynchronizer('x', x_ref, 'y')

        n_cores = 2
        n_evals = 50
        eval_dict = {'a':np.random.uniform(0, 1, n_evals),
                      'b':np.random.uniform(0, 1, n_evals)}

        study = mc.ParameterStudy(pc)
        study.add_evaluation_set(model, obj)
        study.set_core_limit(n_cores)
        study.set_use_threads()
        for i_eval in range(n_evals):
            a_val = eval_dict['a'][i_eval]
            b_val = eval_dict['b'][i_eval]
            study.add_parameter_evaluation(a=a_val, b=b_val)

        base_start = time.time()
        results = study.launch()
        base_end = time.time()
        base_time_delta = base_end - base_start
        study2 = mc.ParameterStudy(pc)
        study2.add_evaluation_set(model, obj)
        study2.set_core_limit(n_cores)
        study2.set_use_threads()
        study2.restart()
        with self.assertRaises(RuntimeError):
            study2.launch()
