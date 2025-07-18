import functools
import matcal as mc
import numpy as np
import time

from matcal.core.models import MatCalSurrogateModel
from matcal.core.surrogates import (SurrogateGenerator, _package_parameter_ranges)
from matcal.core.tests.MatcalUnitTest import MatcalUnitTest


def my_function( n_pts=25, **param_dict):
    import numpy as np
    x = np.linspace(0, 1, n_pts)
    y = param_dict['a'] * x - param_dict['b'] * x
    return {"x": x, "y": y}


def terminate_decorator(func):
    @functools.wraps(func)
    def run_or_not(*args, **kwargs):
        run_or_not.calls += 1
        if run_or_not.calls >= 48:
            raise RuntimeError("no run")
        else:
            return func(*args, **kwargs)
    run_or_not.calls = 0
    return run_or_not


@terminate_decorator
def my_function_term(**param_dict):
    time.sleep(.1)
    return my_function(**param_dict)


def my_linear_function(**param_dict):
    import numpy as np
    x = np.linspace(0, 1, 20)
    y = param_dict['a'] * x + param_dict['b']
    return {"x":x, "y":y}


def get_goal_data():
    return mc.convert_dictionary_to_data(my_function(a=2, b=0, n_pts=5))


class PythonSensitivities(MatcalUnitTest):

    def setUp(self):
        super().setUp(__file__)

    def test_pearson(self):
        slope = mc.Parameter("a", -5, 5., distribution="uniform_uncertain")
        intercept = mc.Parameter("b", -5, 5, distribution="uniform_uncertain")
        parameter_collection = mc.ParameterCollection("two_parameter", slope, intercept)

        curve_data = get_goal_data()

        model = mc.PythonModel(my_function)

        study = mc.LhsSensitivityStudy(parameter_collection)
        objective = mc.CurveBasedInterpolatedObjective("x", "y")
        study.add_evaluation_set(model, objective, curve_data)
        study.set_core_limit(6)
        study.set_number_of_samples(6)
        study.set_random_seed(8555)

        results = study.launch()
        goal  = {'a':np.array([np.nan] +4*[1], dtype=float), 'b':np.array([np.nan]+4*[-1],
                                                                           dtype=float)}
        self.assert_close_arrays(results.outcome["pearson:a"][1:], goal['a'][1:])
        self.assert_close_arrays(results.outcome["pearson:b"][1:], goal['b'][1:])

    def test_pearson_restart_distributed(self):
        slope = mc.Parameter("a", -5, 5., distribution="uniform_uncertain")
        intercept = mc.Parameter("b", -5, 5, distribution="uniform_uncertain")
        parameter_collection = mc.ParameterCollection("two_parameter", slope, intercept)

        curve_data = get_goal_data()
        model = mc.PythonModel(my_function_term)
        model.set_name('py_model')
        
        study = mc.LhsSensitivityStudy(parameter_collection)
        objective = mc.CurveBasedInterpolatedObjective("x", "y")
        study.add_evaluation_set(model, objective, curve_data)
        study.set_number_of_samples(50)
        study.set_random_seed(8555)

        base_start = time.time()
        try:
            base_results = study.launch()
        except Exception:
            pass
        base_delta = time.time() - base_start
        
        # Garbage collection is necessary because h5py close is not fully releasing
        # the lock on the hdf5 file. Calling garbage collection fixes this problem. 
        # Calling collect in the close method does not seem to work. However, because
        # This is really only a problem for automated testing this is okay. 
        import gc
        gc.collect()

        model = mc.PythonModel(my_function)
        model.set_name('py_model')

        study = mc.LhsSensitivityStudy(parameter_collection)
        objective = mc.CurveBasedInterpolatedObjective("x", "y")
        study.add_evaluation_set(model, objective, curve_data)
        study.set_number_of_samples(50)
        study.set_random_seed(8555)
        study.restart()

        restart_start = time.time()
        restart_results = study.launch()
        restart_delta = time.time() - restart_start

        time_ratio = base_delta / restart_delta
        min_ratio = 3
        print(time_ratio)
        self.assertGreaterEqual(time_ratio, min_ratio)

        goal  = {'a':np.array([np.nan] +4*[1], dtype=float), 'b':np.array([np.nan]+4*[-1],
                                                                           dtype=float)}
        self.assert_close_arrays(restart_results.outcome["pearson:a"][1:], goal['a'][1:])
        self.assert_close_arrays(restart_results.outcome["pearson:b"][1:], goal['b'][1:])

    def test_pearson_restart_serial(self):
        slope = mc.Parameter("a", -5, 5., distribution="uniform_uncertain")
        intercept = mc.Parameter("b", -5, 5, distribution="uniform_uncertain")
        parameter_collection = mc.ParameterCollection("two_parameter", slope, intercept)

        curve_data = get_goal_data()
        model = mc.PythonModel(my_function_term)
        model.set_name('py_model')
        
        study = mc.LhsSensitivityStudy(parameter_collection)
        objective = mc.CurveBasedInterpolatedObjective("x", "y")
        study.add_evaluation_set(model, objective, curve_data)
        study.set_number_of_samples(50)
        study.set_random_seed(8555)
        study.run_in_serial()

        base_start = time.time()
        try:
            base_results = study.launch()
        except Exception:
            pass
        base_delta = time.time() - base_start
        
        # Garbage collection is necessary becuase h5py close is not fully releaseing
        # the lock on the hdf5 file. Calling garbage collection fixes this problem. 
        import gc
        gc.collect()

        model = mc.PythonModel(my_function)
        model.set_name('py_model')

        study = mc.LhsSensitivityStudy(parameter_collection)
        objective = mc.CurveBasedInterpolatedObjective("x", "y")
        study.add_evaluation_set(model, objective, curve_data)
        study.set_number_of_samples(50)
        study.set_random_seed(8555)
        study.restart()
        study.run_in_serial()

        restart_start = time.time()
        restart_results = study.launch()
        restart_delta = time.time() - restart_start

        time_ratio = base_delta / restart_delta
        min_ratio = 3
        self.assertGreaterEqual(time_ratio, min_ratio)

        goal  = {'a':np.array([np.nan] +4*[1], dtype=float), 'b':np.array([np.nan]+4*[-1],
                                                                           dtype=float)}
        self.assert_close_arrays(restart_results.outcome["pearson:a"][1:], goal['a'][1:])
        self.assert_close_arrays(restart_results.outcome["pearson:b"][1:], goal['b'][1:])

    def test_pearson_multiple_objective_overall(self):
        state_1 = mc.State("s1", s=1)
        state_2 = mc.State("s2", s=-1)

        def linear_model_func(a,s):
            import numpy as np
            x = np.linspace(0,5,5)
            y=s*a*x
            w = -s*a*x
            return {'x':x, 'y':y, 'w':w}

        def exponential_model_func(a,s):
            import numpy as np
            x = np.linspace(1, 5,5)
            z=np.exp(x**(a*s))

            return {'x':x, 'z':z}

        a = mc.Parameter("a", 0, 4, distribution="uniform_uncertain")
     
        lin_data1  = mc.convert_dictionary_to_data(linear_model_func(a.get_current_value(), 
                                                                     state_1.params["s"]))
        lin_data1.set_state(state_1)
        lin_data2  = mc.convert_dictionary_to_data(linear_model_func(a.get_current_value(), 
                                                                     state_2.params["s"]))
        lin_data2.set_state(state_2)
        lin_data_collection = mc.DataCollection("exp_data", lin_data1, lin_data2)

        exp_data1  = mc.convert_dictionary_to_data(exponential_model_func(a.get_current_value(), 
                                                                          state_1.params["s"]))
        exp_data1.set_state(state_1)
        exp_data2  = mc.convert_dictionary_to_data(exponential_model_func(a.get_current_value(), 
                                                                          state_2.params["s"]))
        exp_data2.set_state(state_2)
        exp_data_collection = mc.DataCollection("exp_data", exp_data1, exp_data2)

        lin_model = mc.PythonModel(linear_model_func)
        exp_model = mc.PythonModel(exponential_model_func)

        study = mc.LhsSensitivityStudy(a)
        objective_lin = mc.CurveBasedInterpolatedObjective("x", "y", "w")
        from matcal.core.objective import L2NormMetricFunction
        objective_lin.set_metric_function(L2NormMetricFunction())
        objective_exp = mc.CurveBasedInterpolatedObjective("x", "z")
        objective_exp.set_metric_function(L2NormMetricFunction())

        study.add_evaluation_set(lin_model, objective_lin, lin_data_collection)
        study.add_evaluation_set(exp_model, objective_exp, exp_data_collection)
        
        study.set_core_limit(6)
        study.set_number_of_samples(20)
        study.set_random_seed(8555)
        study.use_overall_objective()
        results = study.launch()
        #kinda just trusting unit testing tests if this is right.
        goal  = np.array([-0.120781,  0.377279]) 
        self.assert_close_arrays(results.outcome["pearson:a"], goal)

    def test_pearson_multiple_objective(self):
        state_1 = mc.State("s1", s=1)
        state_2 = mc.State("s2", s=-1)

        def linear_model_func(a,s):
            import numpy as np
            x = np.linspace(0,5,5)
            y=s*a*x
            w = -s*a*x
            return {'x':x, 'y':y, 'w':w}


        def exponential_model_func(a,s):
            import numpy as np
            x = np.linspace(1, 5,5)
            z=np.exp(x**(a*s))

            return {'x':x, 'z':z}

        a = mc.Parameter("a", 0, 4, distribution="uniform_uncertain")
     
        lin_data1  = mc.convert_dictionary_to_data(linear_model_func(a.get_current_value(), 
                                                                     state_1.params["s"]))
        lin_data1.set_state(state_1)
        lin_data2  = mc.convert_dictionary_to_data(linear_model_func(a.get_current_value(),
                                                                      state_2.params["s"]))
        lin_data2.set_state(state_2)
        lin_data_collection = mc.DataCollection("exp_data", lin_data1, lin_data2)

        exp_data1  = mc.convert_dictionary_to_data(exponential_model_func(a.get_current_value(), 
                                                                          state_1.params["s"]))
        exp_data1.set_state(state_1)
        exp_data2  = mc.convert_dictionary_to_data(exponential_model_func(a.get_current_value(),
                                                                           state_2.params["s"]))
        exp_data2.set_state(state_2)
        exp_data_collection = mc.DataCollection("exp_data", exp_data1, exp_data2)

        lin_model = mc.PythonModel(linear_model_func)
        exp_model = mc.PythonModel(exponential_model_func)

        study = mc.LhsSensitivityStudy(a)
        objective_lin = mc.CurveBasedInterpolatedObjective("x", "y", "w")
        objective_exp = mc.CurveBasedInterpolatedObjective("x", "z")
        
        study.add_evaluation_set(lin_model, objective_lin, lin_data_collection)
        study.add_evaluation_set(exp_model, objective_exp, exp_data_collection)
        
        study.set_core_limit(6)
        study.set_number_of_samples(20)
        study.set_random_seed(8555)
        results = study.launch()
        goal  = np.array([ np.nan,  1.      ,  1.      ,  1.      ,  1.      , #linear model y state 1, should be 1
                           np.nan, -1.      , -1.      , -1.      , -1.      , #linear model y state 2, should be -1
                           np.nan, -1.      , -1.      , -1.      , -1.      ,  #linear model w state 1, should be -1
                           np.nan,  1.      ,  1.      ,  1.      ,  1.      , #linear model w state 2, should be 1
                           np.nan,  0.43193 ,  0.370024,  0.370024,  0.      ,  #exponential model z state 1, should be positive
                           np.nan, -0.906623, -0.838098, -0.794986, -0.765526]) #exponential model z state 1, should be negative

        self.assert_close_arrays(results.outcome["pearson:a"], goal)

    def test_overall_pearson(self):
        slope = mc.Parameter("a", 0, 5, distribution="uniform_uncertain")
        intercept = mc.Parameter("b", 0, 5, distribution="uniform_uncertain")
        parameter_collection = mc.ParameterCollection("two_parameter", slope, intercept)

        n_pts = 10
        x = np.linspace(0, 1, n_pts)
        y = np.zeros(n_pts)
        data_dict = {'x':x, 'y':y}
        curve_data = mc.convert_dictionary_to_data(data_dict)

        model = mc.PythonModel(my_linear_function)

        study = mc.LhsSensitivityStudy(parameter_collection)
        study.use_overall_objective()
        objective = mc.CurveBasedInterpolatedObjective("x", "y")
        from matcal.core.objective import L2NormMetricFunction
        objective.set_metric_function(L2NormMetricFunction())
        study.add_evaluation_set(model, objective, curve_data)
        study.set_core_limit(6)
        study.set_number_of_samples(20)
        study.set_random_seed(8555)

        results = study.launch()
        goal  = {'a':0.999596, 'b':0.99987}
        self.assert_close_arrays(results.outcome["pearson:a"], goal['a'])
        self.assert_close_arrays(results.outcome["pearson:b"], goal['b'])

    def test_sobol(self):
        slope = mc.Parameter("a", -10, 10., distribution="uniform_uncertain")
        intercept = mc.Parameter("b", -5, 5, distribution="uniform_uncertain")
        parameter_collection = mc.ParameterCollection("two_parameter", slope, intercept)

        curve_data = get_goal_data()

        model = mc.PythonModel(my_function)

        study = mc.LhsSensitivityStudy(parameter_collection)
        study.make_sobol_index_study()
        objective = mc.CurveBasedInterpolatedObjective("x", "y")

        study.add_evaluation_set(model, objective, curve_data)
        study.set_core_limit(6)
        study.set_number_of_samples(12) # not converged
        study.set_random_seed(8555)

        results = study.launch()
        goal = {'a': np.array([[0.58002449, 0.67104569],
       [0.58002449, 0.67104569],
       [0.58002449, 0.67104569],
       [0.58002449, 0.67104569]]), 'b': np.array([[0.10938528, 0.18619558],
       [0.10938528, 0.18619558],
       [0.10938528, 0.18619558],
       [0.10938528, 0.18619558]])}
        self.assert_close_arrays(results.outcome["sobol:a"], goal['a'])
        self.assert_close_arrays(results.outcome["sobol:b"], goal['b'])

    def test_no_data_sim_res_synch(self):
        slope = mc.Parameter("a", 1, 3, distribution="uniform_uncertain")
        intercept = mc.Parameter("b", -1, 1, distribution="uniform_uncertain")
        parameter_collection = mc.ParameterCollection("two_parameter", slope, intercept)

        curve_data = get_goal_data()

        model = mc.PythonModel(my_function)

        study = mc.LhsSensitivityStudy(parameter_collection)
        study.use_overall_objective()
        objective = mc.SimulationResultsSynchronizer("x", curve_data["x"], "y")

        study.add_evaluation_set(model, objective)
        study.set_core_limit(6)
        study.set_number_of_samples(20) # not converged
        study.set_random_seed(8555)

        results = study.launch()
        goal  = {'a':1.0, 'b':-1.0}
        self.assert_close_arrays(results.outcome["pearson:a"], goal['a'])
        self.assert_close_arrays(results.outcome["pearson:b"], goal['b'])


def exp_function(n_pts=100, **parameters):
    # solution to dY/dt = aY
    time, Y0, A = _extract(n_pts, parameters)
    y = Y0 * np.exp(A * time)
    return {"time":time, "y":y}


def _extract(n_pts, parameters):
    time = np.linspace(0, 10, n_pts)
    A = parameters["A"]
    B = parameters['B']
    return time, A, B


def line(n_pts=100, **parameters):
    time, A, B = _extract(n_pts, parameters)
    y = time * A + B
    return {"time":time, "y":y}


def line_sine(n_pts=100, **parameters):
    time, A, B = _extract(n_pts, parameters)
    y = time * A + B * np.sin(time)
    return {"time":time, "y":y}


def multi_line2(n_pts=100, **parameters):
    time = np.linspace(0, 10, n_pts)
    m1 = parameters['m1']
    m2 = parameters['m2']
    xt = 4
    y = time * m1 + np.multiply((time - xt) * m2, time > xt)
    return {"time":time, "y":y}


def multi_line3(n_pts=100, **parameters):
    time = np.linspace(0, 10, n_pts)
    m1 = parameters['m1']
    m2 = parameters['m2']
    xt = parameters['xt']
    y = time * m1 + np.multiply((time - xt) * m2, time > xt)
    return {"time":time, "y":y}


def fake_stress_strain(n_pts=100, **parameters):
    time = np.linspace(0, 1, n_pts)
    E = parameters['E']
    Y = parameters['Y']
    n = parameters['n']
    stress = E * np.power(time, n)
    stress[time < Y] = E * time[time < Y]
    stress[time >= Y] = E * Y + E * (time[time >= Y] - Y)
    return {'time': time, "y":stress}
    

def function_with_localization(n_pts = 100, **parameters):
    time = np.linspace(0, 10, n_pts)
    x = parameters['x']
    y = parameters['y']
    locs = np.hstack([np.atleast_1d(x), np.atleast_1d(y)])
    
    def global_function(positions):
        min_location = np.array([3, 4])
        delta = np.divide(positions - min_location, min_location)
        scale = 1 / 2.91
        dist = np.sum(np.power(delta, 2))
        z = scale * dist
        return z
    
    def local_function(positions):
        min_location = np.array([2, 3])
        decay_length = np.array([1, 1]) / 2
        scale = -2
        z_score = np.divide((positions - min_location), decay_length)
        dist = np.sum(np.power(z_score, 2))
        z = scale * np.exp(-dist)
        return z
    
    z = global_function(locs) + local_function(locs)
    
    return {'time': time, 'z':z *np.ones_like(time)}


class TestLHSToSurrogate(MatcalUnitTest):

    def setUp(self):
        super().setUp(__file__)

    def test_two_parameters_line_sine(self):
        A = mc.Parameter("A", 0, 5, distribution="uniform_uncertain")
        B = mc.Parameter("B", -2, 2, distribution="uniform_uncertain")
        parameter_collection = mc.ParameterCollection("two_parameter", A, B)
        my_function = line_sine
        n_interp = 90
        n_test = 10
        n_source = 500
        sur_type = "Gaussian Process"

        test_points, surrogate = self._use_lhs_to_generate_surrogate(parameter_collection, 
                                                                     my_function, n_interp, 
                                                                     n_test,n_source,  
                                                                     sur_type)
        
        for a, b in zip(test_points['A'], test_points["B"]):
            sur_guess = surrogate([a, b])
            truth = my_function(n_pts = 200, **{"A":a, "B":b})
            self.assert_close_arrays(truth['y'], sur_guess['y'], rtol=5e-2, 
                                     atol=1e-2, show_on_fail=True)

    def test_fake_elastic_plastic(self):
        E = mc.Parameter("E", 1, 5, distribution="uniform_uncertain")
        Y = mc.Parameter("Y", .08, .15, distribution="uniform_uncertain")
        n = mc.Parameter("n", .01, .5, distribution="uniform_uncertain")
        parameter_collection = mc.ParameterCollection("two_parameter", E, Y, n)
        my_function = fake_stress_strain
        n_interp = 90
        n_test = 10
        n_source = 1000
        sur_type = "Gaussian Process"

        test_points, surrogate = self._use_lhs_to_generate_surrogate(parameter_collection,
                                                                      my_function, 
                                                                      n_interp, 
                                                                      n_test, n_source, 
                                                                      sur_type, 1.0)
        
        for e, y, n in zip(test_points['E'], test_points['Y'], test_points['n']):
            sur_guess = surrogate([e, y, n])
            truth = my_function(n_pts = 200, **{"E":e, "Y":y, 'n':n})
            self.assert_close_arrays(truth['y'], sur_guess['y'], rtol=5e-2, atol=1e-2, 
                                     show_on_fail=True)

    def _run_surrogate_calibration(self, param_coll, target_data, hifi_model, 
                                   objective, lhs_save, n_samples):
        lhs = self._run_sampling(param_coll, target_data, hifi_model,
                                 objective, lhs_save, n_samples)
        
        sur_gen = SurrogateGenerator(lhs)
        surrogate = sur_gen.generate('my_surrogate')
        
        sur_model = MatCalSurrogateModel(surrogate)
        
        cal = mc.GradientCalibrationStudy(param_coll)
        cal.add_evaluation_set(sur_model, objective, target_data)
        cal.set_core_limit(15)
        
        cal_results = cal.launch()
        return cal_results

    def _run_sampling(self, param_coll, target_data, hifi_model, objective, lhs_save, n_samples):
        lhs = mc.LhsSensitivityStudy(param_coll)
        lhs.add_evaluation_set(hifi_model, objective, target_data)
        lhs.set_core_limit(15)
        lhs.set_number_of_samples(n_samples)
        lhs.set_seed(12345)
        lhs_results = lhs.launch()
        return lhs

    def _use_lhs_to_generate_surrogate(self, parameter_collection, my_function,
                                        n_interp, n_test, n_source, sur_type, time_end=10.):
        
        demo_data = mc.convert_dictionary_to_data({'time':np.linspace(0, 
                                                                      time_end, n_interp), 
                                                                      'y':np.zeros(n_interp)})

        model = mc.PythonModel(my_function)

        study = mc.LhsSensitivityStudy(parameter_collection)
        objective = mc.CurveBasedInterpolatedObjective("time", "y")
        study.add_evaluation_set(model, objective, demo_data)
        study.set_core_limit(14)
        study.set_number_of_samples(n_source)
        study.set_seed(12345)
        results = study.launch()
        
        sur_gen = SurrogateGenerator(study, 'time')
        sur_gen.set_surrogate_details("PCA Multiple Regressors", sur_type)
        sur_gen.set_PCA_details(decomp_var=5)
        surrogate = sur_gen.generate('my_surrogate')
        test_points = self._make_test_points(results.parameter_history, n_test)

        return test_points,surrogate

    def _make_test_points(self, param_history, n_test):
        p_range = _package_parameter_ranges(param_history)
        test_points = {}
        np.random.seed(12345)
        for name, vals in p_range.items():
            test_points[name] = np.random.uniform(vals[0], vals[1], n_test)
        return test_points