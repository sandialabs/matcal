from collections import OrderedDict
import numpy as np
from scipy.stats import qmc
import unittest

from matcal.core.data import convert_dictionary_to_data, DataCollection
from matcal.core.logger import matcal_print_message
from matcal.core.parameters import ParameterCollection
from matcal.core.serializer_wrapper import matcal_save
from matcal.core.study_base import StudyResults
from matcal.core.surrogates import (_MatCalSurrogateWrapper, 
                                    load_matcal_surrogate, SurrogateGenerator)
from matcal.core.tests.MatcalUnitTest import MatcalUnitTest


def _setup_initial_surrogate_generator(n_samples, p_names, p_low, p_high, 
                                       indep_var, test_function, **parameter_mod):
    p_hist = _generate_parameter_hist_lhs(p_names, p_low, p_high, n_samples)
    for param_name, mod_func in parameter_mod.items():
        if param_name in p_hist.keys():
            p_hist[param_name] = mod_func(p_hist[param_name])
    res_hist, model_name = _generate_parameter_evaluations(test_function, p_hist, p_names)
    res = StudyResults()
    res._update_parameter_history(p_hist, list(p_hist.keys()))

    res._update_simulation_history(res_hist, model_name)
    matcal_save("test_surrogate_source_data.joblib", res)
    sur_gen = SurrogateGenerator(res, indep_var)
    
    return sur_gen


def _generate_parameter_hist_lhs(names, low, high, n_samples):
        params = OrderedDict()
        pc = ParameterCollection('test')
        lhs = qmc.LatinHypercube(d=len(names), seed=10)
        unit_samples = lhs.random(n_samples)
        samples = qmc.scale(unit_samples, low, high)
        for idx in range(n_samples):
            params[f"eval_{idx}"] = OrderedDict()
            for n_idx, param_name in enumerate(names):
                params[f"eval_{idx}"][param_name] = samples[idx, n_idx]
        return params 


def _generate_parameter_evaluations(function, params_hist, param_order):
    results_hist = DataCollection("test model results history")
    for eval in params_hist:
        params = params_hist[eval]
        fun_results = function(**params)
        results_hist.add(convert_dictionary_to_data(fun_results))
    model_name = "simple surrogate"
    return results_hist, model_name

class TestSurrogateGenerator(MatcalUnitTest):

    def setUp(self):
        super().setUp(__file__)

    def test_surrogate_add_generate_preprocessor(self):
        n_samples = 500
        params = {"m":(0, 1), "b":(-1, 1)}
        p_names = ['m', 'b']
        p_low = [0, -1]
        p_high = [1, 1]
        show_array = True
        probes = ['y']
        indep_var = 'x'
        res_file = "test_results"
        err_tol = 1e-2
        n_interp = 200
        interp_locations = np.linspace(0, 10, n_interp)

        def test_function(m, b, n_features=None):
            if n_features == None:
                n_features = np.random.randint(10, 50)
            x = np.linspace(0, 10, n_features)
            y = m * x + b
            return {'x':x, 'y':y}
        
        def preprocessor_func(data):
            for field in list(data.keys()):
                if field != "x":
                    data[field] *= 2.0
            return data

        sur_gen = _setup_initial_surrogate_generator(n_samples, p_names, 
                                                     p_low, p_high, indep_var, test_function)
        sur_gen.set_surrogate_details("PCA Multiple Regressors", "Gaussian Process")
        surrogate = sur_gen.generate('my_surrogate', preprocessing_function=preprocessor_func)

        self._confirm_alignment_to_function(p_low, p_high, show_array, probes, 
                                            err_tol, n_interp, test_function, surrogate, 2)
        self._confirm_good_test_scores(surrogate)
        with self.assertRaises(TypeError):        
            sur_gen.generate('my_surrogate', preprocessing_function="not_func")
            
    def _confirm_good_test_scores(self, surrogate):
        for field in surrogate.scores['test']:
            worst_scores = surrogate.scores['test'][field]['min']
            if isinstance(worst_scores, (float, int)):
                self.assertGreaterEqual(worst_scores, 0.99)
            else:
                for idx in range(len(worst_scores)):
                    self.assertGreaterEqual(worst_scores[idx], 0.99)

    def _confirm_alignment_to_function(self, p_low, p_high, show_array, probes, 
                                       err_tol, n_interp, test_function, surrogate, 
                                       scale_factor=1, log_indices=[]):
        test_sets = self._make_test_sets_uniform(p_low, p_high, log_indices)
        self._assert_passes_fraction_of_times(test_function, show_array, probes, 
                                              err_tol, n_interp, surrogate, 
                                              test_sets, scale_factor)

    def test_surrogate_for_line(self):
        def test_function(m, b, n_features=None):
            if n_features == None:
                n_features = np.random.randint(10, 50)
            x = np.linspace(0, 10, n_features)
            y = m * x + b
            return {'x':x, 'y':y}

        n_samples = 500
        p_names = ['m', 'b']
        p_low = [0, -1]
        p_high = [1, 1]
        show_array = True
        probes = ['y']
        indep_var = 'x'
        res_file = "test_results"
        err_tol = 1e-2
        n_interp = 200

        sur_gen = _setup_initial_surrogate_generator(n_samples, p_names, p_low, p_high, 
                                                     indep_var, test_function)
        sur_gen.set_surrogate_details("PCA Multiple Regressors", "Gaussian Process")
        surrogate = sur_gen.generate('my_surrogate')

        self._confirm_alignment_to_function(p_low, p_high, show_array, probes, err_tol, n_interp, 
                                            test_function, surrogate)
        self._confirm_good_test_scores(surrogate)

    @unittest.skip("Fails with log scaling. Need to fix.")
    def test_surrogate_for_log_scale_parameter(self):
        def test_function(m, b, n_features=None):
            if n_features == None:
                n_features = np.random.randint(50, 150)
            x = np.linspace(0, 10, n_features)
            y = b * np.exp(-x * m)
            return {'x':x, 'y':y}

        n_samples = 500 
        p_names = ['m', 'b']
        p_low = [-2, 1]
        p_high = [-.5, 4]
        show_array = True
        probes = ['y']
        indep_var = 'x'
        err_tol = 1e-2
        n_interp = 200

        def raise_10_to(p):
            return np.power(10, p)

        sur_gen = _setup_initial_surrogate_generator(n_samples, p_names, p_low, p_high, indep_var, 
                                                     test_function, m=raise_10_to)
        sur_gen.set_surrogate_details("PCA Multiple Regressors", "Gaussian Process", alpha=1.e-6)
        sur_gen.set_fields_to_log_scale('y', 'm')
        sur_gen.set_PCA_details(None, 1e-2)
        surrogate = sur_gen.generate('my_surrogate')

        self._confirm_alignment_to_function(p_low, p_high, show_array, probes, err_tol, n_interp, 
                                            test_function, surrogate, log_indices=[0])
        self._confirm_good_test_scores(surrogate)

    def test_surrogate_for_line_integer_num_components(self):
        def test_function(m, b, n_features=None):
            if n_features == None:
                n_features = np.random.randint(10, 50)
            x = np.linspace(0, 10, n_features)
            y = m * x + b
            return {'x':x, 'y':y}

        n_samples = 500
        p_names = ['m', 'b']
        p_low = [0, -1]
        p_high = [1, 1]
        show_array = True
        probes = ['y']
        indep_var = 'x'
        res_file = "test_results"
        err_tol = 1e-2
        n_interp = 200
        
        sur_gen = _setup_initial_surrogate_generator(n_samples, p_names, p_low, p_high, 
                                                     indep_var, test_function)
        sur_gen.set_surrogate_details("PCA Multiple Regressors", "Gaussian Process")
        sur_gen.set_PCA_details(decomp_var=2)
        surrogate = sur_gen.generate('my_surrogate')

        self._confirm_alignment_to_function(p_low, p_high, show_array, probes, err_tol, 
                                            n_interp, test_function, surrogate)
        self._confirm_good_test_scores(surrogate)

    def test_surrogate_for_line_string_num_components(self):
        def test_function(m, b, n_features=None):
            if n_features == None:
                n_features = np.random.randint(10, 50)
            x = np.linspace(0, 10, n_features)
            y = m * x + b
            return {'x':x, 'y':y}

        n_samples = 500
        p_names = ['m', 'b']
        p_low = [0, -1]
        p_high = [1, 1]
        show_array = True
        probes = ['y']
        indep_var = 'x'
        err_tol = 1e-2
        n_interp = 200

        sur_gen = _setup_initial_surrogate_generator(n_samples, p_names, p_low, p_high, 
                                                     indep_var, test_function)
        sur_gen.set_surrogate_details("PCA Multiple Regressors", "Gaussian Process")
        sur_gen.set_PCA_details(decomp_var='mle')
        surrogate = sur_gen.generate('my_surrogate')

        self._confirm_alignment_to_function(p_low, p_high, show_array, probes, err_tol, 
                                            n_interp, test_function, surrogate)
        self._confirm_good_test_scores(surrogate)

    def test_surrogate_for_line_string_few_features_skip_PCA(self):
        def test_function(m, b, n_features=5):
            if n_features == None:
                n_features = np.random.randint(10, 50)
            x = np.linspace(0, 10, n_features)
            y = m * x + b
            return {'x':x, 'y':y}

        n_samples = 500
        p_names = ['m', 'b']
        p_low = [0, -1]
        p_high = [1, 1]
        show_array = True
        probes = ['y']
        indep_var = None
        err_tol = 1e-2
        n_interp = 5

        sur_gen = _setup_initial_surrogate_generator(n_samples, p_names, p_low, p_high, 
                                                     indep_var, test_function)
        sur_gen.set_surrogate_details("PCA Multiple Regressors", "Gaussian Process")
        surrogate = sur_gen.generate('my_surrogate')
        self._confirm_alignment_to_function(p_low, p_high, show_array, probes, err_tol, 
                                            n_interp, test_function, surrogate)
        self._confirm_good_test_scores(surrogate)


    def test_surrogate_for_line_read_from_file(self):
        def test_function(m, b, n_features=None):
            if n_features == None:
                n_features = np.random.randint(10, 50)
            x = np.linspace(0, 10, n_features)
            y = m * x + b
            return {'x':x, 'y':y}

        n_samples = 500
        p_names = ['m', 'b']
        p_low = [0, -1]
        p_high = [1, 1]
        show_array = True
        probes = ['y']
        indep_var = 'x'
        res_file = "test_results"
        err_tol = 1e-2
        n_interp = 200
        interp_locations = np.linspace(0, 10, n_interp)
        
        sur_gen = _setup_initial_surrogate_generator(n_samples, p_names, p_low, 
                                                     p_high, indep_var, test_function)
        sur_gen.set_surrogate_details("PCA Multiple Regressors", "Gaussian Process")
        sur_gen.set_PCA_details(decomp_var='mle')
        sur_gen.generate('my_surrogate')
        surrogate_loaded = load_matcal_surrogate('my_surrogate.joblib')

        self._confirm_alignment_to_function(p_low, p_high, show_array, probes, err_tol, 
                                            n_interp, test_function, surrogate_loaded)
        self._confirm_good_test_scores(surrogate_loaded)
        
        
    def test_surrogate_for_line_monolythic(self):
        def test_function(m, b, n_features=None):
            if n_features == None:
                n_features = np.random.randint(10, 50)
            x = np.linspace(0, 10, n_features)
            y = m * x + b
            return {'x':x, 'y':y}

        n_samples = 500
        p_names = ['m', 'b']
        p_low = [0, -1]
        p_high = [1, 1]
        show_array = True
        probes = ['y']
        indep_var = 'x'
        err_tol = 1e-2
        n_interp = 200
        
        sur_gen = _setup_initial_surrogate_generator(n_samples, p_names, p_low, 
                                                     p_high, indep_var, test_function)
        sur_gen.set_surrogate_details("PCA Monolythic Regressor", "Gaussian Process")
        surrogate = sur_gen.generate('my_surrogate')

        self._confirm_alignment_to_function(p_low, p_high, show_array, probes, 
                                            err_tol, n_interp, test_function, surrogate)
        self._confirm_good_test_scores(surrogate)
        

    def test_surrogate_for_line_read_from_file_monolythic(self):
        def test_function(m, b, n_features=None):
            if n_features == None:
                n_features = np.random.randint(10, 50)
            x = np.linspace(0, 10, n_features)
            y = m * x + b
            return {'x':x, 'y':y}

        n_samples = 500
        p_names = ['m', 'b']
        p_low = [0, -1]
        p_high = [1, 1]
        show_array = True
        probes = ['y']
        indep_var = 'x'
        res_file = "test_results"
        err_tol = 1e-2
        n_interp = 200
        interp_locations = np.linspace(0, 10, n_interp)

        sur_gen = _setup_initial_surrogate_generator(n_samples, p_names, 
                                                     p_low, p_high, indep_var, test_function)
        sur_gen.set_surrogate_details("PCA Monolythic Regressor", "Gaussian Process")
        sur_gen.generate('my_surrogate')
        surrogate = load_matcal_surrogate("my_surrogate.joblib")

        self._confirm_alignment_to_function(p_low, p_high, show_array, probes, 
                                            err_tol, n_interp, test_function, surrogate)
        self._confirm_good_test_scores(surrogate)


    def test_surrogate_for_constant_random_forest(self):
        def test_function(b, n_features=None):
            if n_features == None:
                n_features = np.random.randint(10, 50)
            x = np.linspace(0, 10, n_features)
            y = b * np.ones_like(x)
            return {'x':x, 'y':y}

        n_samples = 1500
        p_names = ['b']
        p_low = [-5]
        p_high = [5]
        show_array = True
        probes = ['y']
        indep_var = 'x'
        res_file = "test_results"
        err_tol = 1e-2
        n_interp = 200
        
        sur_gen = _setup_initial_surrogate_generator(n_samples, p_names, 
                                                     p_low, p_high, indep_var, test_function)
        sur_gen.set_surrogate_details("PCA Monolythic Regressor", "Random Forest")
        surrogate = sur_gen.generate('my_surrogate')

        self._confirm_alignment_to_function(p_low, p_high, show_array, 
                                            probes, err_tol, n_interp, test_function, surrogate)
        self._confirm_good_test_scores(surrogate)
    
    def test_surrogate_for_2_lines(self):
        def test_function(m, b, n_features=None):
            if n_features == None:
                n_features = np.random.randint(50, 100)
            x = np.linspace(0, 10, n_features)
            y = m * x + b
            z = b * x + m
            return {'x':x, 'y':y, 'z':z}

        n_samples = 250
        p_names = ['m', 'b']
        p_low = [0, 4]
        p_high = [3, 10]
 
        show_array = True
        probes = ['y', 'z']
        indep_var = 'x'
        res_file = "test_results"
        err_tol = 1e-2

        n_interp = 200
        sur_gen = _setup_initial_surrogate_generator(n_samples, p_names, p_low, 
                                                     p_high, indep_var, test_function)
        sur_gen.set_surrogate_details("PCA Multiple Regressors", "Gaussian Process")
        surrogate = sur_gen.generate('my_surrogate')

        self._confirm_alignment_to_function(p_low, p_high, show_array, probes, 
                                            err_tol, n_interp, test_function, surrogate)
        self._confirm_good_test_scores(surrogate)
        
    def test_surrogate_for_decay(self):
        time_end = 10
        def test_function(L, A, n_features=None):
            if n_features == None:
                n_features = np.random.randint(75, 150)
            x = np.linspace(0, time_end, n_features)
            l_eff = np.power(10, L)
            y = np.exp(-x * l_eff) * A + 1
            return {'x':x, 'y':y}

        n_samples = 200
        p_names = ['L', 'A']
        p_low = [-1., 1]
        p_high = [0, 10]
        show_array = True
        probes = ['y']
        indep_var = 'x'
        res_file = "test_results"
        err_tol = 1e-2

        n_interp = 200
        sur_gen = _setup_initial_surrogate_generator(n_samples, p_names, p_low, 
                                                     p_high, indep_var, test_function)
        sur_gen.set_surrogate_details("PCA Multiple Regressors", "Gaussian Process")
        sur_gen.set_PCA_details(None, reconstruction_error=1e-2)
        surrogate = sur_gen.generate('my_surrogate')

        self._confirm_alignment_to_function(p_low, p_high, show_array, probes, 
                                            err_tol, n_interp, test_function, surrogate)
        self._confirm_good_test_scores(surrogate)

    def test_surrogate_for_cos_and_line_var_based(self):
        time_end = 10
        def test_function(A, n_features=None):
            if n_features == None:
                n_features = np.random.randint(75, 150)
            x = np.linspace(0, time_end, n_features)
            y = np.cos(x / 2) * A 
            z = A * x + A
            return {'x':x, 'y':y, 'z':z}

        n_samples = 100
        p_names = ['A']
        p_low = [0]
        p_high = [2]
        show_array = True
        probes = ['y', 'z']
        indep_var = 'x'
        res_file = "test_results"
        err_tol = 1e-2

        n_interp = 200
        sur_gen = _setup_initial_surrogate_generator(n_samples, p_names, p_low, 
                                                     p_high, indep_var, test_function)
        sur_gen.set_surrogate_details("PCA Multiple Regressors", "Gaussian Process")
        surrogate = sur_gen.generate('my_surrogate')

        self._confirm_alignment_to_function(p_low, p_high, show_array, probes, 
                                            err_tol, n_interp, test_function, surrogate)
        self._confirm_good_test_scores(surrogate)

    def test_surrogate_reconstruction_based(self):
        time_end = 10
        def test_function(A, B, n_features=None):
            if n_features == None:
                n_features = np.random.randint(200, 300)
            x = np.linspace(0, time_end, n_features)
            y = np.cos(x / 10) * A + B * np.exp(-x)
            z = A * x + A + np.power(x, 1/B)
            return {'x':x, 'y':y, 'z':z}

        n_samples = 400
        p_names = ['A', "B"]
        p_low = [0, 1]
        p_high = [2, 2]
        show_array = True
        probes = ['y', 'z']
        indep_var = 'x'
        res_file = "test_results"
        err_tol = 1e-2

        n_interp = 200
        sur_gen = _setup_initial_surrogate_generator(n_samples, p_names, p_low, 
                                                     p_high, indep_var, test_function)
        sur_gen.set_surrogate_details("PCA Multiple Regressors", "Gaussian Process", alpha=1e-6)
        sur_gen.set_PCA_details(None, reconstruction_error=5e-3)
        surrogate = sur_gen.generate('my_surrogate')

        self._confirm_alignment_to_function(p_low, p_high, show_array, probes, 
                                            err_tol, n_interp, test_function, surrogate)
        self._confirm_good_test_scores(surrogate)


    def test_surrogate_wrapper(self):
        time_end = 10
        def test_function(A, n_features=None):
            if n_features == None:
                n_features = np.random.randint(75, 150)
            x = np.linspace(0, time_end, n_features)
            y = np.cos(x / 2) * A 
            z = A * x + A
            return {'x':x, 'y':y, 'z':z}

        n_samples = 200
        p_names = ['A']
        p_low = [0]
        p_high = [2]
        show_array = True
        probes = ['y', 'z']
        indep_var = 'x'
        res_file = "test_results"
        err_tol = 1e-4

        n_interp = 200
        sur_gen = _setup_initial_surrogate_generator(n_samples, p_names, p_low, 
                                                     p_high, indep_var, test_function)
        sur_gen.set_surrogate_details("PCA Multiple Regressors", 
                                      "Gaussian Process", alpha=1e-6)
        sur_gen.set_PCA_details(None, reconstruction_error=5e-3)
        surrogate = sur_gen.generate('my_surrogate')
        python_model_like = _MatCalSurrogateWrapper(surrogate)
        tp = {'A':1.2}
        prediction = python_model_like(**tp)
        goal = test_function(tp['A'], n_interp)
        self.assert_close_arrays(prediction['y'], goal['y'], rtol=err_tol, 
                                 show_on_fail=show_array)
        self.assert_close_arrays(prediction['z'], goal['z'], rtol=err_tol, 
                                 show_on_fail=show_array)
        
    def _assert_passes_fraction_of_times(self, test_function, show_array, 
                                         probes, err_tol, n_interp, 
                                         surrogate, test_sets, 
                                         goal_scale_factor =1.0):
        N_passed = 0
        N_failed = 0
        passed_record = []
        error_record = []
        for test_set in test_sets:
            for test_field in probes:
                goal = test_function(*test_set, n_interp)[test_field]*goal_scale_factor
                prediction = surrogate(np.array(test_set).reshape(1, -1))[test_field]
                results = self.check_if_close_arrays(prediction, 
                                                     goal, 
                                                     1e-4, err_tol, 
                                                     False, 
                                                     show_array)
                passed_set,processed_first_array, processed_second_array = results
                delta = np.abs(processed_first_array-processed_second_array)
                max_delta = np.max(delta)
                location_of_max = np.argmax(delta)
                passed_record.append(passed_set)
                error_record.append(max_delta)
                if passed_set:
                    N_passed += 1
                else:
                    N_failed += 1
        passed = N_passed / (N_failed + N_passed) > .9
        if not passed:
            matcal_print_message("Num passed:", N_passed)
            matcal_print_message("Num failed:", N_failed)
            out_data = {'passed':np.array(passed_record), 'error':np.array(error_record)}
            
            for i_var in range(len(test_sets[0])):
                record = []
                for test_set in test_sets:
                    record.append(test_set[i_var])
                out_data[f'var_{i_var}'] = np.array(record)
            matcal_save('passed_failed_parameters.joblib', out_data)
        self.assertTrue(N_passed / (N_failed + N_passed) > .9)

    def _make_test_sets_uniform(self, low, high, log_indices=[]):
        test_sets = []
        n_sets = 50
        for set_i in range(n_sets):
            cur_set = []
            for i in range(len(low)):
                new_value = np.random.uniform(low[i], high[i])
                if i in log_indices:
                    new_value = np.power(10, new_value)
                cur_set.append(new_value)
            test_sets.append(cur_set)
        return test_sets

    def _make_test_sets_normal(self, means, stds):
        test_sets = []
        std_mults = [0, 1, -1]
        for m0 in std_mults:
            for m1 in std_mults:
                cur_set = []
                m = [m0, m1]
                for i in range(len(means)):
                    cur_set.append(means[i] + m[i] * stds[i])
                test_sets.append(cur_set)
        return test_sets