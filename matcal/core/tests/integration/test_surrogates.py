import numpy as np
import unittest

from matcal.core.logger import matcal_print_message
from matcal.core.serializer_wrapper import matcal_save, matcal_load
from matcal.core.surrogates import (_MatCalSurrogateWrapper)
from matcal.core.tests.unit.test_surrogates import (_setup_initial_surrogate_generator, 
                                                    TestSurrogateGenerator)


class TestSurrogateGenerator(TestSurrogateGenerator):

    def setUp(self):
        super().setUp()
      
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
            return {'y':y}

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
        surrogate_loaded = matcal_load('my_surrogate.joblib')

        self._confirm_alignment_to_function(p_low, p_high, show_array, probes, err_tol, 
                                            n_interp, test_function, surrogate_loaded)
        self._confirm_good_test_scores(surrogate_loaded)
        
    def test_surrogate_for_line_monolithic(self):
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
        sur_gen.set_surrogate_details("PCA Monolithic Regressor", "Gaussian Process")
        surrogate = sur_gen.generate('my_surrogate')

        self._confirm_alignment_to_function(p_low, p_high, show_array, probes, 
                                            err_tol, n_interp, test_function, surrogate)
        self._confirm_good_test_scores(surrogate)

    def test_surrogate_for_line_read_from_file_monolithic(self):
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
        sur_gen.set_surrogate_details("PCA Monolithic Regressor", "Gaussian Process")
        sur_gen.generate('my_surrogate')
        surrogate = matcal_load("my_surrogate.joblib")

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
        sur_gen.set_surrogate_details("PCA Monolithic Regressor", "Random Forest")
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

    def test_surrogate_for_2_lines_only_one_field_of_interest(self):
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
        probes = ['y']
        indep_var = 'x'
        res_file = "test_results"
        err_tol = 1e-2

        n_interp = 200
        sur_gen = _setup_initial_surrogate_generator(n_samples, p_names, p_low, 
                                                     p_high, indep_var, test_function)
        sur_gen.set_surrogate_details("PCA Multiple Regressors", "Gaussian Process")
        sur_gen.set_fields_of_interest("y")
        surrogate = sur_gen.generate('my_surrogate')
        
        self._confirm_alignment_to_function(p_low, p_high, show_array, probes, 
                                            err_tol, n_interp, test_function, surrogate)
        self._confirm_good_test_scores(surrogate)

    def test_surrogate_for_decay(self):
        time_end = 10
        def test_function(L, A, n_features=None):
            if n_features is None:
                n_features = np.random.randint(75, 150)
            x = np.linspace(0, time_end, n_features)
            l_eff = np.power(10, L)
            y = np.exp(-x * l_eff) * A + 1
            return {'x': x, 'y': y}

        n_samples = 500
        p_names = ['L', 'A']
        p_low = [-1., 1]
        p_high = [0, 10]
        show_array = True
        probes = ['y']
        indep_var = 'x'
        err_tol = 1e-2

        n_interp = 200
        sur_gen = _setup_initial_surrogate_generator(n_samples, p_names, p_low,
                                                     p_high, indep_var, test_function)
        # alpha provides numerical stability for GP on variable-length output
        sur_gen.set_surrogate_details("PCA Multiple Regressors", "Gaussian Process",
                                      alpha=1e-6)
        sur_gen.set_PCA_details(None, reconstruction_error=1e-2)
        surrogate = sur_gen.generate('my_surrogate')

        self._confirm_alignment_to_function(p_low, p_high, show_array, probes,
                                            err_tol, n_interp, test_function, surrogate)
        self._confirm_good_test_scores(surrogate)

    def test_surrogate_for_cos_and_line_var_based(self):
        time_end = 10
        def test_function(A, n_features=None):
            if n_features is None:
                n_features = np.random.randint(75, 150)
            x = np.linspace(0, time_end, n_features)
            y = np.cos(x / 2) * A
            z = A * x + A
            return {'x': x, 'y': y, 'z': z}

        n_samples = 400
        p_names = ['A']
        p_low = [0]
        p_high = [2]
        show_array = True
        probes = ['y', 'z']
        indep_var = 'x'
        err_tol = 1e-2

        n_interp = 200
        sur_gen = _setup_initial_surrogate_generator(n_samples, p_names, p_low,
                                                     p_high, indep_var, test_function)
        # alpha provides numerical stability for the GP on this variable-length problem
        sur_gen.set_surrogate_details("PCA Multiple Regressors", "Gaussian Process",
                                      alpha=1e-6)
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

        n_samples = 1000
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
                                                     p_high, indep_var, test_function, 
                                                     interp_locations=n_interp)
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

        n_samples = 500
        p_names = ['A']
        p_low = [0]
        p_high = [2]
        show_array = True
        probes = ['y', 'z']
        indep_var = 'x'
        res_file = "test_results"
        err_tol = 3e-4

        n_interp = 200
        sur_gen = _setup_initial_surrogate_generator(n_samples, p_names, p_low, 
                                                     p_high, indep_var, test_function, 
                                                     interp_locations=n_interp)
        sur_gen.set_surrogate_details("PCA Multiple Regressors", 
                                      "Gaussian Process", alpha=1e-6)
        sur_gen.set_PCA_details(None, reconstruction_error=5e-3)
        surrogate = sur_gen.generate('my_surrogate')
        python_model_like = surrogate
        tp = {'A':1.2}
        prediction = python_model_like(**tp)
        goal = test_function(tp['A'], n_interp)
        self.assert_close_arrays(prediction['y'], goal['y'], rtol=err_tol, 
                                 show_on_fail=show_array)
        self.assert_close_arrays(prediction['z'], goal['z'], rtol=err_tol, 
                                 show_on_fail=show_array)
    
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