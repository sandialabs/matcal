
import numpy as np
from scipy.stats import qmc
from sklearn.discriminant_analysis import StandardScaler
import unittest

from matcal.core.data import (convert_dictionary_to_data, _serialize_data)
from matcal.core.parameters import ParameterCollection, Parameter
from matcal.core.parameter_studies import ParameterStudy
from matcal.core.serializer_wrapper import _format_serial, matcal_save
from matcal.core.study_base import StudyResults
from matcal.core.surrogates import (_ReconstructionDecomposition, 
                                    _VarianceDecomposition, MatCalLogScaler, 
                                    _assign_decomp, 
                                    _identify_fields_of_interest, 
                                    _import_and_interpolate, _import_parameter_hist, 
                                    _make_parameter_scaler_set,  
                                    _package_parameter_ranges, _parse_evaluation_info, 
                                    _process_training_data, _process_interpolation_locations, 
                                    _scale_data_for_surrogate, _score_recreation,
                                    _tune_data_decomposition, 
                                    _WorstEvaluations, _select_state_data, _select_model, 
                                    _apply_preprocessing_function)

from matcal.core.tests.MatcalUnitTest import MatcalUnitTest
from matcal.core.tests.utilities_for_tests import _generate_singe_model_single_state_mock_eval_hist


def _generate_study_results_lhs(n_evals, parameter_dict, eval_function):
    n_param = len(parameter_dict)
    study_results = StudyResults()
    eval_set = 'one_set'
    study_results._evaluation_sets = [eval_set]
    _generate_sample_history(n_evals, parameter_dict, n_param, study_results)
    _populate_simulation_data(n_evals, eval_function, study_results, eval_set)
    return study_results

    
def _generate_sample_history(n_evals, parameter_dict, n_param, study_results):
    low, high = _get_param_limits(parameter_dict)
    lhs = qmc.LatinHypercube(d=n_param, seed=10)
    unit_samples = lhs.random(n_evals)
    samples = qmc.scale(unit_samples, low, high)
    for p_idx, p_name in enumerate(parameter_dict):
        study_results._parameter_history[p_name] = samples[:, p_idx]


def _get_param_limits(parameter_dict):
    low = []
    high = []
    for cur_low, cur_high in parameter_dict.values():
        low.append(cur_low)
        high.append(cur_high)
    return low,high


class TestSurrogateFunctions(MatcalUnitTest):

    def setUp(self):
        super().setUp(__file__)

    def test_parse_study_results_returns_parameters_and_qois(self):
        p_names = ['a', 'b']
        p_means = [0, 1]
        p_stds = [.1, .2]
        n_samples = 10
        
        def qoi_fun(a, b):
            n_pts = 4
            time = np.linspace(0, 1, n_pts)
            return {'time':time, 'c': time * (a+b)}
        
        sr = _generate_singe_model_single_state_mock_eval_hist(p_names, p_means,
                                                               p_stds, n_samples, 
                                                               qoi_fun)
        out_goal = sr.simulation_history["MockModel"]
        in_hist, out_hist = _parse_evaluation_info(sr, None)
        self.assert_close_dicts_or_data(in_hist, sr.parameter_history)
        self.assertEqual(out_hist, out_goal)

    def test_parse_evaluation_from_study_results(self):
        p_names = ['a', 'b']
        p_means = [0, 1]
        p_stds = [.1, .2]
        n_samples = 10
        
        def qoi_fun(a, b):
            n_pts = 4
            time = np.linspace(0, 1, n_pts)
            return {'time':time, 'c': time * (a+b)}
        
        sr = _generate_singe_model_single_state_mock_eval_hist(p_names, p_means,
                                                               p_stds, n_samples, 
                                                               qoi_fun)
        
        out_goal = sr.simulation_history["MockModel"]
        in_hist, out_hist = _parse_evaluation_info(sr, None)
        self.assert_close_dicts_or_data(in_hist, sr.parameter_history)
        self.assertEqual(out_hist, out_goal)        

    def test_identify_fields_of_interest(self):
        indep_field = 'time'
        n_pts = 10
        data_list = [convert_dictionary_to_data({'time':np.linspace(0, 2, n_pts), 
                       'c':np.ones(n_pts), 
                       'd':np.ones(n_pts)}), 
                       convert_dictionary_to_data({'time':np.linspace(1, 2, n_pts), 
                       'c':np.ones(n_pts), 
                       'd':np.ones(n_pts)})]
        goal = ['c', 'd']
        foi = _identify_fields_of_interest(data_list, indep_field)
        self.assertEqual(len(goal), len(foi))
        for name in goal: 
            self.assertIn(name, foi)

    def test_parse_evaluation_from_study(self):
        p_names = ['a', 'b']
        p_means = [0, 1]
        p_stds = [.1, .2]
        n_samples = 10
        
        def qoi_fun(a, b):
            n_pts = 4
            time = np.linspace(0, 1, n_pts)
            return {'time':time, 'c': time * (a+b)}
        
        sr = _generate_singe_model_single_state_mock_eval_hist(p_names, p_means,
                                                               p_stds, n_samples, 
                                                               qoi_fun)
        
        study_stub = ParameterStudy(Parameter('a', 0, 1), Parameter('b', 0, 1))
        study_stub._results = sr

        out_goal = sr.simulation_history
        out_goal = out_goal["MockModel"]
        in_hist, out_hist = _parse_evaluation_info(study_stub, None)
        self.assert_close_dicts_or_data(in_hist, sr.parameter_history)
        self.assertEqual(out_hist, out_goal)    

    def test_process_interpolation_locations_passed_array_accept_array(self):
        interp_field = 'time'
        passed_interp = np.linspace(0, 1, 100)
        output_hist = None
        interp_locations = _process_interpolation_locations(output_hist, passed_interp, interp_field)
        self.assert_close_arrays(interp_locations, passed_interp)

    def test_process_interpolation_location_auto_generates_array_spaning_fully_supported_region(self):
        interp_field = 'time'
        passed_interp = 200
        goal_start = 5
        goal_end = 10
        data_list = [convert_dictionary_to_data({'time':np.linspace(0, goal_end, 100), 
                       'a':np.ones(100), }), 
                       convert_dictionary_to_data({'time':np.linspace(goal_start, 12, 80), 
                       'a': np.zeros(80)})]

        interp_locations = _process_interpolation_locations(data_list, passed_interp, interp_field)
        goal = np.linspace(goal_start, goal_end, 200)
        self.assert_close_arrays(interp_locations, goal)

    def test_parse_evaluation_from_passed_dict(self):
        p_names = ['a', 'b']
        p_means = [0, 1]
        p_stds = [.1, .2]
        n_samples = 10
        
        def qoi_fun(a, b):
            n_pts = 4
            time = np.linspace(0, 1, n_pts)
            return {'time':time, 'c': time * (a+b)}
        
        sr = _generate_singe_model_single_state_mock_eval_hist(p_names, p_means,
                                                               p_stds, n_samples, 
                                                               qoi_fun)

        first_eval = sr.evaluation_sets[0]
        out_goal = sr.simulation_history['MockModel']
        my_dict = {'input':sr.parameter_history, 
                   'output':sr.simulation_history['MockModel']}      
        in_hist, out_hist = _parse_evaluation_info(my_dict, None)  
        self.assert_close_dicts_or_data(in_hist, sr.parameter_history)
        self.assertEqual(out_goal, out_hist) 

    def test_import_and_interpolate_exists(self):
        search_string = "test_file*.json"
        interp_locations = np.linspace(0, 1, 5)
        interp_field = 'time'
        fields_of_interest = ['T', 'U']
        imported_dict = _import_and_interpolate(search_string, fields_of_interest,
                                                 interp_field, interp_locations)

    def test_process_data_corpus_no_preprocessor_only_interp(self):
        param_names = ['a', 'b', 'c']
        param_mean = [1, -3, 5]
        param_std = [.2, 0, 1.5]
        param_var = np.power(param_std, 2)
        n_eval = 300
        
        def test_fun(a,b,c):
            n = 4
            return {'z':np.linspace(0, 4, n) * a *b *c, 'time': np.linspace(0, 10, n)}
        
        interp_times = np.linspace(0, 10, 10)
        interp_mult = np.linspace(0, 4, 10)
        
        study_results = _generate_singe_model_single_state_mock_eval_hist(param_names, param_mean, param_std, n_eval, test_fun) 
        p_hist = study_results.parameter_history
        qois_dc = study_results.simulation_history["MockModel"]
        qois = _select_state_data(None, qois_dc)
        goal_z = np.outer(np.multiply(p_hist['a'], np.multiply(p_hist['b'], p_hist['c'])), interp_mult)
        process_data = _process_training_data(qois, ['z'], interp_times, 'time')
        self.assert_close_arrays(goal_z, process_data['z'], show_on_fail=True)
  
    def test_process_data_corpus_interp_and_preprocess(self):
        param_names = ['a', 'b', 'c']
        param_mean = [1, -3, 5]
        param_std = [.2, 0, 1.5]
        n_eval = 30
        
        def test_fun(a,b,c):
            n = 18
            return {'z':np.linspace(0, 4, n)*a*b*c, 'time': np.linspace(0, 10, n)}
        
        def preprocess_function(data):
            data['z'] /= 2
            return data
        
        interp_times = np.linspace(0, 10, 10)
        interp_mult = np.linspace(0, 4, 10) / 2
        
        study_results = _generate_singe_model_single_state_mock_eval_hist(param_names, param_mean, 
                                                                          param_std, n_eval,
                                                                          test_fun) 
        p_hist = study_results.parameter_history
        goal_z = np.outer(np.multiply(p_hist['a'], np.multiply(p_hist['b'], p_hist['c'])), 
                          interp_mult)
        eval_set_name = study_results.evaluation_sets[0]
        model = _select_model(study_results.simulation_history, None)
        qois_dc = study_results.simulation_history[model]
        qois = _select_state_data("MockState", qois_dc)

        process_data = _apply_preprocessing_function(preprocess_function, qois)
        process_data = _process_training_data(qois, ['z'], interp_times, 'time')
        self.assert_close_arrays(goal_z, process_data['z'], show_on_fail=True)

    def test_process_data_corpus_interp_and_preprocess_return_dict(self):
        param_names = ['a', 'b', 'c']
        param_mean = [1, -3, 5]
        param_std = [.2, 0, 1.5]
        n_eval = 30
        
        def test_fun(a,b,c):
            n = 18
            return {'z':np.linspace(0, 4, n)*a*b*c, 'time': np.linspace(0, 10, n)}
        
        def preprocess_function(data):
            data['z'] /= 2
            return {'z':data['z'], 'time':data['time']}
        
        interp_times = np.linspace(0, 10, 10)
        interp_mult = np.linspace(0, 4, 10) / 2
        
        study_results = _generate_singe_model_single_state_mock_eval_hist(param_names, param_mean, 
                                                                          param_std, n_eval,
                                                                          test_fun) 
        p_hist = study_results.parameter_history
        goal_z = np.outer(np.multiply(p_hist['a'], np.multiply(p_hist['b'], p_hist['c'])), 
                          interp_mult)
        eval_set_name = study_results.evaluation_sets[0]
        model = _select_model(study_results.simulation_history, None)
        qois_dc = study_results.simulation_history[model]
        qois = _select_state_data("MockState", qois_dc)

        process_data = _apply_preprocessing_function(preprocess_function, qois)
        process_data = _process_training_data(qois, ['z'], interp_times, 'time')
        self.assert_close_arrays(goal_z, process_data['z'], show_on_fail=True)

    def test_import_and_interpolate_returns_same_values_for_same_points(self):
        search_string = "test_file*.json"
        interp_locations = np.linspace(0, 1, 5)
        interp_field = 'time'
        fields_of_interest = ['T', 'U']
        time = interp_locations
        T = 3 * time + 5
        U = np.exp(time)
        goal_dict = {"time":time, "T":T, "U":U}
        goal_data = convert_dictionary_to_data(goal_dict)
        test_filename = "test_file0.json"
        matcal_save(test_filename, _serialize_data(goal_data))             
        imported_dict = _import_and_interpolate(search_string, fields_of_interest, 
                                                interp_field, interp_locations)
        self.assertEqual(len(imported_dict), len(fields_of_interest))
        for field in fields_of_interest:
            self.assert_close_arrays(imported_dict[field], goal_data[field])

    def test_import_and_interpolate_returns_same_values_for_same_pattern_different_points(self):
        search_string = "test_file*.json"
        n_time = 5
        interp_locations = np.linspace(0, 1, n_time)
        interp_field = 'time'
        fields_of_interest = ['T', 'U']
        
        T_goal = 3 * interp_locations + 5
        U_goal = np.exp(interp_locations)
        goal_dict = {"time":interp_locations, "T":T_goal, "U":U_goal}

        time = np.linspace(0, 1, 3*n_time)
        T = 3 * time + 5
        U = np.exp(time)
        example_dict = {"time":time, "T":T, "U":U}

        example_data = convert_dictionary_to_data(example_dict)
        test_filename = "test_file0.json"
        matcal_save(test_filename, _serialize_data(example_data))     
               
        imported_dict = _import_and_interpolate(search_string, fields_of_interest,
                                                 interp_field, interp_locations)
        self.assertEqual(len(imported_dict), len(fields_of_interest))
        for field in fields_of_interest:
            self.assert_close_arrays(imported_dict[field], goal_dict[field])
    
    def test_import_and_interpolate_returns_same_values_for_same_pattern_different_points_n_files(self):
        search_string = "test_file*.json"
        n_time = 5
        interp_locations = np.linspace(0, 1, n_time)
        interp_field = 'time'
        fields_of_interest = ['T', 'U']
        
        T_goal = 3 * interp_locations + 5
        U_goal = np.exp(interp_locations)
        goal_dict = {"time":interp_locations, "T":T_goal, "U":U_goal}

        n_files = 7
        for i in range(n_files):
            time = np.linspace(0, 1, 3*np.random.randint(n_time, 10*n_time))
            T = 3 * time + 5
            U = np.exp(time)
            example_dict = {"time":time, "T":T, "U":U}

            example_data = convert_dictionary_to_data(example_dict)
            test_filename = f"test_file{i}.json"
            matcal_save(test_filename, _serialize_data(example_data))     
               
        imported_dict = _import_and_interpolate(search_string, fields_of_interest, 
                                                interp_field, interp_locations)
        self.assertEqual(len(imported_dict), len(fields_of_interest))
        for field in fields_of_interest:
            for f_i in range(n_files):
                self.assert_close_arrays(imported_dict[field][f_i, :], goal_dict[field])

    def test_scaling_all_same_value(self):
        n_samp = 4
        n_feat = 5
        one_array = np.ones([n_samp, n_feat])
        goal_array = np.zeros_like(one_array)
        goal_mean = np.ones(n_feat)
        goal_std = np.zeros(n_feat)
        scaled_array, scaling_object = _scale_data_for_surrogate(one_array)
        self.assert_close_arrays(goal_array, scaled_array)
        self.assert_close_arrays(scaling_object.mean_, goal_mean)
        self.assert_close_arrays(scaling_object.var_, goal_std)
        
    def test_do_log_scaling(self):
        n_samp = 4
        n_feat = 5
        low_val = 0
        source_array = np.random.uniform(low_val, 1, [n_samp, n_feat])
        source_array[0,:] = low_val
        goal_array = StandardScaler().fit_transform(np.log10(source_array - low_val + 1))
        make_log = True
        scaled_array, scaling_object = _scale_data_for_surrogate(source_array, make_log)
        self.assert_close_arrays(goal_array, scaled_array)

        

    def test_scaling_line_with_white_noise(self):
        n_samp = 100000
        n_feat = 10
        time = np.linspace(0, 1, n_feat)

        std = 1.25
        noise = np.random.normal(0, std, (n_samp, n_feat))
        signal = 4 + 2 * time
        data = noise + signal
        scaled_array, scaling_object = _scale_data_for_surrogate(data)
        self.assert_close_arrays(scaling_object.mean_, signal, atol=5e-2)
        self.assert_close_arrays(scaling_object.var_, 
                                 np.ones(n_feat) * np.power(std, 2),
                                 atol=5e-2)

    def test_parameter_history_generate_scaled_parameters(self):
        param_names = ['a', 'b', 'c']
        param_mean = [1, -3, 5]
        param_std = [.2, 0, 1.5]
        param_var = np.power(param_std, 2)
        n_eval = 30
        
        input_params = {}
        for name, mu, s in zip(param_names, param_mean, param_std):
            input_params[name] = np.random.normal(mu, s, n_eval)

        imported_parameters = _import_parameter_hist(input_params)
        fields_to_log_scale = []
        pss = _make_parameter_scaler_set(imported_parameters, fields_to_log_scale)
        scaled_parameters = pss.transform_to_array(imported_parameters)
        parameter_key_order = pss.parameter_order
        
        self.assert_close_arrays(scaled_parameters.shape, [n_eval, len(param_names)])
        self.assert_close_arrays(scaled_parameters[:, 1], 0)

        # An okay, test this is showing that all values are within 4 std of the mean.
        # This may mean there are some rare test failure because the source points have 
        # an outlier. 
        self.assertTrue((scaled_parameters <= 4.0).all() and (scaled_parameters >= -4.0).all())

        self.assertEqual(len(param_names), len(parameter_key_order))
        for name in param_names:
            self.assertIn(name, parameter_key_order)

    def test_package_parameter_ranges(self):
        pc = {"A":np.linspace(0, 1), "Z":np.linspace(-4, 20)}
        ppr = _package_parameter_ranges(pc)
        self.assertIn("A", ppr.keys())
        self.assertIn("Z", ppr.keys())
        self.assert_close_arrays(ppr["A"], (0, 1))
        self.assert_close_arrays(ppr["Z"], (-4, 20))
       
    def _make_fake_eval_hist(obj_array, param_dict, model_names):
        obj_history = {}
        for model_name in model_names:
            obj_history[model_name] = []
        
    def _gen_obj_hist(self, obj_func, samples):
        obj_evals = obj_func(samples)
        obj_hist = []
        for values in list(obj_evals):
            cur_obj = {}
            for obj_idx, value in enumerate(np.atleast_1d(values)):
                cur_obj[f'obj_{obj_idx}'] = {'objective':value}
            cur_eval ={'fake_model':cur_obj}
            obj_hist.append(cur_eval)
        return obj_hist

    def _make_samples_lhs(self, names, low, high, n_samples):
        lhs = qmc.LatinHypercube(d=len(names), seed=10)
        unit_samples = lhs.random(n_samples)
        samples = qmc.scale(unit_samples, low, high)
        return samples
        
    def _make_samples_uniform(self, names, low, high, n_samples):
        n_dim = len(names)
        n_axis = int(np.ceil(np.power(n_samples, 1/n_dim)))
        revised_n_samples = int(np.power(n_axis, n_dim))
        axis = []
        for i_dim in range(n_dim):
            axis.append(np.linspace(low[i_dim], high[i_dim], n_axis))
        grids = np.meshgrid(*axis)
        samples = np.zeros([n_dim, revised_n_samples])
        for i_dim in range(n_dim):
            samples[i_dim, :] = grids[i_dim].flatten()
        return samples.T
    
    
    class _DataGenerator:
        
        def __init__(self, func, *nominal_args):
            self._func = func
            self._n_args = np.array(nominal_args)
            
        def __call__(self, n_iter, delta_fraction):
            delta = self._n_args *  delta_fraction
            low = self._n_args - delta
            high = self._n_args + delta
            results = []
            params = []
            for cur_iter in range(n_iter):
                cur_param = np.random.uniform(low, high)
                params.append(cur_param)
                cur_res = self._func(*list(cur_param))
                results.append(cur_res)
            return np.array(params), np.array(results)


    def test_convert_data_and_make_bias_tuner(self):
        recreation_error_tolerance = 1e-3
        def my_fun(a, b):
            x = np.linspace(0, 1, 100)
            y = a * np.power(x, b)
            return y
        
        my_data_generator = self._DataGenerator(my_fun, 1, 1)
        train_params, train_source_data = my_data_generator(100, .25)
        test_params, test_source_data = my_data_generator(20, .25)
        data_scaler, decomposer, scaled_latent_data, latent_scaler = _tune_data_decomposition(train_source_data, recreation_error_tolerance)
        
        scaled_test_data = data_scaler.transform(test_source_data)
        test_latent_data = decomposer.transform(scaled_test_data)
        recreated_scaled_test_data = decomposer.inverse_transform(test_latent_data)
        recreated_test_data = data_scaler.inverse_transform(recreated_scaled_test_data)
        self.assert_close_arrays(test_source_data, recreated_test_data, atol=recreation_error_tolerance, show_arrays=True)
        
        
        
    def test_worst_evaluations_collector_have_nothing_at_start(self):
        n_track = 2
        we = _WorstEvaluations(n_track)
        self.assertEqual(len(we.get_set()), 0)
        
    def test_worst_evaluations_collector_store_2(self):
        n_track = 2
        we = _WorstEvaluations(n_track)
        
        # Add 1 Eval
        field_0 = 'a'
        eval_idx_0 = 3
        score_0 = 10
        we.update(field_0, eval_idx_0, score_0)
        
        worst_set = we.get_set()
        self.assertEqual(len(worst_set),1)
        self._assert_correct_terms_at_index(field_0, eval_idx_0, worst_set, 0)

        # Add 2nd eval better added in order of addition      
        field_1 = 'b'
        eval_idx_1 = 123
        score_1 = 4
        we.update(field_1, eval_idx_1, score_1)
        
        worst_set = we.get_set()
        self.assertEqual(len(worst_set),2)
        self._assert_correct_terms_at_index(field_0, eval_idx_0, worst_set, 0)
        self._assert_correct_terms_at_index(field_1, eval_idx_1, worst_set, 1)
        
        # add 3ed set that boots 2nd, and now sorted by ascending score
        field_2 = 'c'
        eval_idx_2 = 1
        score_2 = 20
        we.update(field_2, eval_idx_2, score_2)
        
        worst_set = we.get_set()
        self.assertEqual(len(worst_set),2)
        self._assert_correct_terms_at_index(field_2, eval_idx_2, worst_set, 1)
        self._assert_correct_terms_at_index(field_0, eval_idx_0, worst_set, 0)

    def _assert_correct_terms_at_index(self, field_0, eval_idx_0, worst_set, query_index):
        r_field, r_idx = worst_set[query_index]
        self.assertEqual(r_field, field_0)
        self.assertEqual(r_idx, eval_idx_0)
        
        
    def test_assign_decomp_if_no_reconstuction_error_get_variance_based(self):
        var_decomp = .9
        recon_error = None
        decomp_tool = _assign_decomp(var_decomp, recon_error)
        self.assertIsInstance(decomp_tool, _VarianceDecomposition)
        
    
    def test_assign_decomp_if_reconstuction_error_get_recon_based(self):
        var_decomp = .9
        recon_error = .1
        decomp_tool = _assign_decomp(var_decomp, recon_error)
        self.assertIsInstance(decomp_tool, _ReconstructionDecomposition)
        
    def test_assign_decomp_raise_error_if_bad_recon_error_tol(self):
        var_decomp = .9
        with self.assertRaises(RuntimeError):
            _assign_decomp(var_decomp, -1)
        with self.assertRaises(RuntimeError):
            _assign_decomp(var_decomp, 1.2)
            
    def test_assign_decomp_raise_error_if_bad_var_tol(self):
        recon_err = None
        with self.assertRaises(RuntimeError):
            _assign_decomp(-1, None)
        with self.assertRaises(RuntimeError):
            _assign_decomp(1.2, None)
            
            
    def test_score_recreation_get_result_scaled_on_reference_data_constant(self):
        n_pts = 10
        ref_values = np.ones(n_pts) * 4
        test_values = np.ones(n_pts) * 2
        goal = np.ones(n_pts) * 2
        goal_score = np.linalg.norm(goal)
        test_score = _score_recreation(test_values, ref_values)
        self.assertAlmostEqual(test_score, goal_score)
        
    def test_log_scaler_make_log10(self):
        scaler = MatCalLogScaler()
        n_eval = 100
        n_param = 2
        data = np.random.uniform(10, 1000, (n_eval, n_param))
        min_value = 10
        data[0, :] = min_value
        scaler.fit(data)
        t_data = scaler.transform(data)
        self.assert_close_arrays(t_data, np.log10(data - min_value + 1))
        
    def test_fit_transform_combo(self):
        scaler = MatCalLogScaler()
        n_eval = 100
        n_param = 2
        data = np.random.uniform(10, 1000, (n_eval, n_param))
        min_value = 10
        data[0, :] = min_value
        t_data = scaler.fit_transform(data)
        self.assert_close_arrays(t_data, np.log10(data - min_value + 1))
        
    def test_inverse_transform_reproduces_original(self):
        scaler = MatCalLogScaler()
        n_eval = 100
        n_param = 2
        data = np.random.uniform(10, 1000, (n_eval, n_param))
        min_value = 10
        data[0, :] = min_value
        t_data = scaler.fit_transform(data)
        i_data = scaler.inverse_transform(t_data)
        self.assert_close_arrays(i_data, data)
                
    
    def test_works_for_negative_numbers(self): 
        scaler = MatCalLogScaler()
        n_eval = 100
        n_param = 1
        min_value = -1000
        data = np.random.uniform(min_value, 1000, (n_eval, n_param))
        data[0, :] = min_value
        t_data = scaler.fit_transform(data)
        self.assert_close_arrays(t_data, np.log10(data - min_value + 1))
        i_data = scaler.inverse_transform(t_data)
        self.assert_close_arrays(i_data, data)           
    
    def test_log_scaler_require_data_dim_greater_than_1(self):
        scaler = MatCalLogScaler()
        n_eval = 100
        data_1d = np.linspace(10, 1000, n_eval)
        data_2d = np.linspace(10, 1000, n_eval).reshape(-1, 1)
        with self.assertRaises(IndexError):
            scaler.fit(data_1d)
        
        with self.assertRaises(IndexError):
            scaler.fit(data_2d)
            scaler.transform(data_1d)
    
    def test_log_scaler_requires_numpy_array(self):
        scaler = MatCalLogScaler()
        n_eval = 100
        data_list = [[1,2,3,4]]
        data_2d = np.linspace(10, 1000, n_eval).reshape(-1, 1)
        with self.assertRaises(TypeError):
            scaler.fit(data_list)
        
        with self.assertRaises(TypeError):
            scaler.fit(data_2d)
            scaler.transform(data_list)