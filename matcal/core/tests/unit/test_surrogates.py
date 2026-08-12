from collections import OrderedDict
import numpy as np
from scipy.stats import qmc
from sklearn.discriminant_analysis import StandardScaler
import types
from unittest.mock import patch


from matcal.core.data import convert_dictionary_to_data, DataCollection
from matcal.core.logger import matcal_print_message
from matcal.core.parameters import Parameter, ParameterCollection
from matcal.core.parameter_studies import ParameterStudy
from matcal.core.serializer_wrapper import matcal_save
from matcal.core.study_base import StudyResults
from matcal.core.surrogates import (
    _DoNothingDataTransformer,
    _MatCalLogScaler, 
    MatCalMonolithicPCASurrogate,
    _ReconstructionDecomposition, 
    _RBFInterpolatorRegressor,
    SurrogateGenerator, 
    _VarianceDecomposition, 
    _WorstEvaluations,
    _apply_preprocessing_function, 
    _apply_regressor_metric,
    _assign_decomp, 
    _calculate_nlpd,
    _calculate_performance_metrics,
    _calculate_response_error_metric,
    _check_fields_in_keys_list,
    _convert_param_array_to_dict,
    _decompose_with_pca,
    _ensure_2d_array,
    _field_uses_pca,
    _get_model_name_from_evaluation_information,
    _get_n_points,
    _identify_fields_of_interest, 
    _import_parameter_hist, 
    _make_parameter_scaler_set,  
    _match_single_column_and_1d_metric_arrays,
    _mean_absolute_error,
    _modal_regressor,
    _normalize_evaluation_information_names,
    _normalized_root_mean_squared_error,
    _package_parameter_ranges, 
    _parse_evaluation_info, 
    _prepare_metric_arrays,
    _print_scores,
    _process_data_for_surrogate, 
    _process_interpolation_locations, 
    _process_surrogate_args_call,
    _record_variance_behaviors,
    _root_mean_squared_error,
    _scale_data_for_surrogate,
    _score_recreation,
    _select_model,
    _select_state_data,  
    _split_qoi_history_key,
    _sum_absolute_error,
    _tune_data_decomposition 
)

from matcal.core.tests.MatcalUnitTest import MatcalUnitTest
from matcal.core.tests.utilities_for_tests import _generate_singe_model_single_state_mock_eval_hist


def _setup_initial_surrogate_generator(n_samples, p_names, p_low, p_high, 
                                       indep_var, test_function, interp_locations=200, training_fraction=0.8,
                                       **parameter_mod):
    test_res = None
    res = _get_parameter_and_simulation_hist(p_names, p_low, p_high, n_samples,
                                             test_function, **parameter_mod)
    matcal_save("test_surrogate_source_data.joblib", res)
    
    if training_fraction == 1.0:
        test_res = _get_parameter_and_simulation_hist(p_names, p_low, p_high, 100,
                                                      test_function, rng=20, **parameter_mod)
        matcal_save("test_surrogate_test_data.joblib", test_res)
    sur_gen = SurrogateGenerator(res, indep_var, training_fraction=training_fraction, 
                                 interpolation_locations=interp_locations,
                                 test_eval_info=test_res)
    
    return sur_gen


def _get_parameter_and_simulation_hist(p_names, p_low, p_high, n_samples, test_function,
                                       rng=10, **parameter_mod):
    p_hist = _generate_parameter_hist_lhs(p_names, p_low, p_high, n_samples, rng=rng)
    for param_name, mod_func in parameter_mod.items():
        if param_name in p_hist.keys():
            p_hist[param_name] = mod_func(p_hist[param_name])
    res_hist, model_name = _generate_parameter_evaluations(test_function, p_hist, p_names)
    res = StudyResults()
    res._update_parameter_history(p_hist, list(p_hist.keys()))

    res._update_simulation_history(res_hist, model_name)
    return res


def _generate_parameter_hist_lhs(names, low, high, n_samples, rng=10):
        params = OrderedDict()
        pc = ParameterCollection('test')
        lhs = qmc.LatinHypercube(d=len(names), seed=rng)
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


def _make_test_sets_uniform(low, high, log_indices=[]):
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


def _make_results_like_for_name_normalization(
    simulation_model_names=("old_model",),
    qoi_keys=("old_model:old_objective",),
):
    simulation_history = OrderedDict()
    for model_name in simulation_model_names:
        simulation_history[model_name] = types.SimpleNamespace(
            state_names=["state"]
        )

    qoi_history = OrderedDict()
    for qoi_key in qoi_keys:
        qoi_history[qoi_key] = object()

    return types.SimpleNamespace(
        simulation_history=simulation_history,
        qoi_history=qoi_history,
    )


class TestSurrogateEvaluationInformationNameNormalization(MatcalUnitTest):

    def setUp(self):
        super().setUp(__file__)

    def test_split_qoi_history_key(self):
        model_name, objective_name = _split_qoi_history_key("model:objective")

        self.assertEqual(model_name, "model")
        self.assertEqual(objective_name, "objective")

    def test_split_qoi_history_key_rejects_invalid_keys(self):
        with self.assertRaises(RuntimeError):
            _split_qoi_history_key("no_colon")

        with self.assertRaises(RuntimeError):
            _split_qoi_history_key(":objective")

        with self.assertRaises(RuntimeError):
            _split_qoi_history_key("model:")

    def test_get_model_name_from_evaluation_information_uses_explicit_model_name(self):
        eval_info = _make_results_like_for_name_normalization(
            simulation_model_names=("model_a", "model_b"),
            qoi_keys=("model_a:objective",),
        )

        model_name = _get_model_name_from_evaluation_information(
            eval_info,
            model_name="explicit_model",
        )

        self.assertEqual(model_name, "explicit_model")

    def test_get_model_name_from_evaluation_information_finds_single_model(self):
        eval_info = _make_results_like_for_name_normalization(
            simulation_model_names=("only_model",),
            qoi_keys=("only_model:objective",),
        )

        model_name = _get_model_name_from_evaluation_information(
            eval_info,
            model_name=None,
        )

        self.assertEqual(model_name, "only_model")

    def test_get_model_name_from_evaluation_information_returns_none_for_multiple_models(self):
        eval_info = _make_results_like_for_name_normalization(
            simulation_model_names=("model_a", "model_b"),
            qoi_keys=("model_a:objective",),
        )

        model_name = _get_model_name_from_evaluation_information(
            eval_info,
            model_name=None,
        )

        self.assertIsNone(model_name)

    def test_normalize_evaluation_information_renames_single_model_and_qoi_names(self):
        results = _make_results_like_for_name_normalization(
            simulation_model_names=("old_model",),
            qoi_keys=("old_model:old_objective",),
        )

        with patch("matcal.core.surrogates.logger.warning") as warning_mock:
            returned = _normalize_evaluation_information_names(
                results,
                required_model_name="required_model",
                required_objective_name="required_objective",
                data_set_name="unit-test data",
                logger_on=True,
            )

        self.assertIs(returned, results)
        self.assertEqual(
            list(results.simulation_history.keys()),
            ["required_model"],
        )
        self.assertEqual(
            list(results.qoi_history.keys()),
            ["required_model:required_objective"],
        )

        self.assertGreaterEqual(warning_mock.call_count, 2)
        warning_text = "\n".join(
            str(call.args[0]) for call in warning_mock.call_args_list
        )
        self.assertIn("old_model", warning_text)
        self.assertIn("required_model", warning_text)
        self.assertIn("old_objective", warning_text)
        self.assertIn("required_objective", warning_text)

    def test_normalize_evaluation_information_noops_when_names_match(self):
        results = _make_results_like_for_name_normalization(
            simulation_model_names=("required_model",),
            qoi_keys=("required_model:required_objective",),
        )

        with patch("matcal.core.surrogates.logger.warning") as warning_mock:
            returned = _normalize_evaluation_information_names(
                results,
                required_model_name="required_model",
                required_objective_name="required_objective",
                data_set_name="unit-test data",
                logger_on=True,
            )

        self.assertIs(returned, results)
        self.assertEqual(
            list(results.simulation_history.keys()),
            ["required_model"],
        )
        self.assertEqual(
            list(results.qoi_history.keys()),
            ["required_model:required_objective"],
        )
        warning_mock.assert_not_called()

    def test_normalize_evaluation_information_errors_for_multiple_simulation_models(self):
        results = _make_results_like_for_name_normalization(
            simulation_model_names=("model_a", "model_b"),
            qoi_keys=("model_a:objective",),
        )

        with self.assertRaises(RuntimeError):
            _normalize_evaluation_information_names(
                results,
                required_model_name="required_model",
                required_objective_name=None,
                data_set_name="unit-test data",
                logger_on=False,
            )

    def test_normalize_evaluation_information_errors_for_multiple_qoi_models(self):
        results = _make_results_like_for_name_normalization(
            simulation_model_names=("required_model",),
            qoi_keys=("model_a:objective", "model_b:objective"),
        )

        with self.assertRaises(RuntimeError):
            _normalize_evaluation_information_names(
                results,
                required_model_name="required_model",
                required_objective_name="required_objective",
                data_set_name="unit-test data",
                logger_on=False,
            )

    def test_normalize_evaluation_information_errors_for_multiple_qoi_objectives(self):
        results = _make_results_like_for_name_normalization(
            simulation_model_names=("required_model",),
            qoi_keys=("required_model:objective_a", "required_model:objective_b"),
        )

        with self.assertRaises(RuntimeError):
            _normalize_evaluation_information_names(
                results,
                required_model_name="required_model",
                required_objective_name="required_objective",
                data_set_name="unit-test data",
                logger_on=False,
            )

    def test_normalize_evaluation_information_allows_multiple_objectives_when_no_required_objective(self):
        results = _make_results_like_for_name_normalization(
            simulation_model_names=("old_model",),
            qoi_keys=("old_model:objective_a", "old_model:objective_b"),
        )

        returned = _normalize_evaluation_information_names(
            results,
            required_model_name="required_model",
            required_objective_name=None,
            data_set_name="unit-test data",
            logger_on=False,
        )

        self.assertIs(returned, results)
        self.assertEqual(
            list(results.simulation_history.keys()),
            ["required_model"],
        )
        self.assertEqual(
            set(results.qoi_history.keys()),
            {
                "required_model:objective_a",
                "required_model:objective_b",
            },
        )

    def test_surrogate_generator_normalizes_test_eval_info_model_name(self):
        train_results = _make_results_like_for_name_normalization(
            simulation_model_names=("training_model",),
            qoi_keys=("training_model:training_objective",),
        )
        test_results = _make_results_like_for_name_normalization(
            simulation_model_names=("test_model",),
            qoi_keys=("test_model:test_objective",),
        )

        generator = SurrogateGenerator.__new__(SurrogateGenerator)
        generator._eval_info = train_results
        generator._test_eval_info = test_results
        generator._model_name = None
        generator._logger_on = True

        with patch("matcal.core.surrogates.logger.warning") as warning_mock:
            generator._normalize_test_evaluation_information_names()

        self.assertEqual(
            list(test_results.simulation_history.keys()),
            ["training_model"],
        )
        self.assertEqual(
            list(test_results.qoi_history.keys()),
            ["training_model:test_objective"],
        )

        self.assertGreaterEqual(warning_mock.call_count, 1)
        warning_text = "\n".join(
            str(call.args[0]) for call in warning_mock.call_args_list
        )
        self.assertIn("test_model", warning_text)
        self.assertIn("training_model", warning_text)

    def test_surrogate_generator_test_eval_info_normalization_noops_without_test_eval_info(self):
        train_results = _make_results_like_for_name_normalization(
            simulation_model_names=("training_model",),
            qoi_keys=("training_model:training_objective",),
        )

        generator = SurrogateGenerator.__new__(SurrogateGenerator)
        generator._eval_info = train_results
        generator._test_eval_info = None
        generator._model_name = None
        generator._logger_on = True

        generator._normalize_test_evaluation_information_names()

        self.assertIsNone(generator._test_eval_info)


class TestSurrogateFunctions(MatcalUnitTest):

    def setUp(self):
        super().setUp(__file__)

    def test_root_mean_squared_error(self):
        test_values = np.array([[1.0, 2.0],
                                [3.0, 4.0]])
        surrogate_values = np.array([[0.0, 2.0],
                                     [5.0, 1.0]])

        expected = np.sqrt(np.mean((test_values - surrogate_values) ** 2))
        actual = _root_mean_squared_error(test_values, surrogate_values)

        self.assertAlmostEqual(actual, expected)

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
        foi = _identify_fields_of_interest(data_list, indep_field, None)
        self.assertEqual(len(goal), len(foi))
        for name in goal: 
            self.assertIn(name, foi)
        foi = _identify_fields_of_interest(data_list, indep_field, ['c'])
        self.assertEqual(foi, ['c'])

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
        process_data = _process_data_for_surrogate(qois, ['z'], interp_times, 'time')
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
        process_data = _process_data_for_surrogate(qois, ['z'], interp_times, 'time')
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
        process_data = _process_data_for_surrogate(qois, ['z'], interp_times, 'time')
        self.assert_close_arrays(goal_z, process_data['z'], show_on_fail=True)

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
        scaled_parameters = pss.transform_as_array(imported_parameters)
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
        scaler = _MatCalLogScaler()
        n_eval = 100
        n_param = 2
        data = np.random.uniform(10, 1000, (n_eval, n_param))
        min_value = 10
        data[0, :] = min_value
        scaler.fit(data)
        t_data = scaler.transform(data)
        self.assert_close_arrays(t_data, np.log10(data - min_value + 1))
        
    def test_fit_transform_combo(self):
        scaler = _MatCalLogScaler()
        n_eval = 100
        n_param = 2
        data = np.random.uniform(10, 1000, (n_eval, n_param))
        min_value = 10
        data[0, :] = min_value
        t_data = scaler.fit_transform(data)
        self.assert_close_arrays(t_data, np.log10(data - min_value + 1))
        
    def test_inverse_transform_reproduces_original(self):
        scaler = _MatCalLogScaler()
        n_eval = 100
        n_param = 2
        data = np.random.uniform(10, 1000, (n_eval, n_param))
        min_value = 10
        data[0, :] = min_value
        t_data = scaler.fit_transform(data)
        i_data = scaler.inverse_transform(t_data)
        self.assert_close_arrays(i_data, data)
    
    def test_works_for_negative_numbers(self): 
        scaler = _MatCalLogScaler()
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
        scaler = _MatCalLogScaler()
        n_eval = 100
        data_1d = np.linspace(10, 1000, n_eval)
        data_2d = np.linspace(10, 1000, n_eval).reshape(-1, 1)
        with self.assertRaises(IndexError):
            scaler.fit(data_1d)
        
        with self.assertRaises(IndexError):
            scaler.fit(data_2d)
            scaler.transform(data_1d)
    
    def test_log_scaler_requires_numpy_array(self):
        scaler = _MatCalLogScaler()
        n_eval = 100
        data_list = [[1,2,3,4]]
        data_2d = np.linspace(10, 1000, n_eval).reshape(-1, 1)
        with self.assertRaises(TypeError):
            scaler.fit(data_list)
        
        with self.assertRaises(TypeError):
            scaler.fit(data_2d)
            scaler.transform(data_list)

    def test_rbf_interpolator_regressor_uses_default_neighbors(self):
        rng = np.random.default_rng(123)
        n_samples = 80
        x = rng.uniform(-1, 1, (n_samples, 2))
        y = x[:, 0] + 2.0 * x[:, 1]

        regressor = _RBFInterpolatorRegressor()
        regressor.fit(x, y)

        self.assertEqual(regressor._effective_neighbors, 20)

        prediction = regressor.predict(x[:5])
        self.assertEqual(prediction.shape, (5,))

        score = regressor.score(x, y)
        self.assertGreater(score, 0.99)

    def test_rbf_interpolator_regressor_respects_user_neighbors(self):
        rng = np.random.default_rng(123)
        n_samples = 40
        x = rng.uniform(-1, 1, (n_samples, 2))
        y = np.sin(x[:, 0]) + x[:, 1] ** 2

        regressor = _RBFInterpolatorRegressor(neighbors=12)
        regressor.fit(x, y)

        self.assertEqual(regressor._effective_neighbors, 12)

    def test_rbf_interpolator_regressor_caps_neighbors_to_training_size(self):
        rng = np.random.default_rng(123)
        n_samples = 10
        x = rng.uniform(-1, 1, (n_samples, 2))
        y = x[:, 0] - x[:, 1]

        regressor = _RBFInterpolatorRegressor(neighbors=50)
        regressor.fit(x, y)

        self.assertEqual(regressor._effective_neighbors, n_samples)

    def test_print_scores_omits_latent_space_scores_when_no_pca_used(self):
        latent_train_score = OrderedDict()
        latent_test_score = OrderedDict()
        native_train_score = OrderedDict()
        native_test_score = OrderedDict()
        decomposers = OrderedDict()

        field = "response"

        latent_train_score[field] = {
            "score": np.array([0.99]),
            "nlpd": np.array([np.nan]),
            "rmse": np.array([0.01]),
        }
        latent_test_score[field] = {
            "score": np.array([0.98]),
            "nlpd": np.array([np.nan]),
            "rmse": np.array([0.02]),
        }

        native_train_score[field] = 0.995
        native_test_score[field] = 0.985

        decomposers[field] = _DoNothingDataTransformer()

        with patch("matcal.core.surrogates.logger.info") as logger_info:
            _print_scores(
                latent_train_score,
                latent_test_score,
                native_train_score,
                native_test_score,
                decomposers=decomposers,
            )

        logged_text = "\n".join(
            str(call.args[0]) for call in logger_info.call_args_list
        )

        self.assertIn("original data space score", logged_text)
        self.assertNotIn("latent space score", logged_text)
        self.assertNotIn("PCA latent space score", logged_text)
        
    def test_print_scores_reports_latent_space_scores_when_pca_used(self):
        latent_train_score = OrderedDict()
        latent_test_score = OrderedDict()
        native_train_score = OrderedDict()
        native_test_score = OrderedDict()
        decomposers = OrderedDict()

        field = "response"

        latent_train_score[field] = {
            "score": np.array([0.99]),
            "nlpd": np.array([np.nan]),
            "rmse": np.array([0.01]),
        }
        latent_test_score[field] = {
            "score": np.array([0.98]),
            "nlpd": np.array([np.nan]),
            "rmse": np.array([0.02]),
        }

        native_train_score[field] = 0.995
        native_test_score[field] = 0.985

        class FakePCA:
            pass

        decomposers[field] = FakePCA()

        with patch("matcal.core.surrogates.logger.info") as logger_info:
            _print_scores(
                latent_train_score,
                latent_test_score,
                native_train_score,
                native_test_score,
                decomposers=decomposers,
            )

        logged_text = "\n".join(
            str(call.args[0]) for call in logger_info.call_args_list
        )

        self.assertIn("original data space score", logged_text)
        self.assertIn("PCA latent space score", logged_text)

    def test_calculate_nlpd_matches_gaussian_formula_with_scalar_std_per_sample(self):

        class FakeGPR:

            def predict(self, input_values, return_std=False):
                mu = np.array([
                    [1.0, 2.0],
                    [3.0, 4.0],
                ])
                std = np.array([0.5, 1.0])

                if return_std:
                    return mu, std

                return mu

        y_true = np.array([
            [1.5, 1.0],
            [2.0, 6.0],
        ])

        mu = np.array([
            [1.0, 2.0],
            [3.0, 4.0],
        ])

        std = np.array([
            [0.5, 0.5],
            [1.0, 1.0],
        ])

        var = std ** 2
        residual = y_true - mu
        expected = 0.5 * np.mean(
            np.log(2.0 * np.pi * var) + residual ** 2 / var
        )

        actual = _calculate_nlpd(
            FakeGPR(),
            input_values=np.zeros((2, 1)),
            y_true=y_true,
        )

        self.assertAlmostEqual(actual, expected)

    def test_calculate_response_error_metric_rmse_reuses_shared_metric(self):
        y_true = np.array([[1.0, 2.0], [3.0, 4.0]])
        y_pred = np.array([[1.0, 1.0], [5.0, 4.0]])

        expected = np.sqrt(np.mean((y_true - y_pred) ** 2))

        actual = _calculate_response_error_metric(y_true, y_pred, "rmse")

        self.assertAlmostEqual(actual, expected)

    def test_calculate_response_error_metric_mae(self):
        y_true = np.array([[1.0, 2.0], [3.0, 4.0]])
        y_pred = np.array([[1.0, 1.0], [5.0, 4.0]])

        expected = np.mean(np.abs(y_true - y_pred))

        actual = _calculate_response_error_metric(y_true, y_pred, "mae")

        self.assertAlmostEqual(actual, expected)

    def test_calculate_response_error_metric_sum_abs(self):
        y_true = np.array([[1.0, 2.0], [3.0, 4.0]])
        y_pred = np.array([[1.0, 1.0], [5.0, 4.0]])

        expected = np.sum(np.abs(y_true - y_pred))

        actual = _calculate_response_error_metric(y_true, y_pred, "sum_abs")

        self.assertAlmostEqual(actual, expected)

    def test_calculate_response_error_metric_nrmse(self):
        y_true = np.array([[1.0, 2.0], [3.0, 4.0]])
        y_pred = np.array([[1.0, 1.0], [5.0, 4.0]])

        expected = np.sqrt(np.sum((y_true - y_pred) ** 2) / np.sum(y_true ** 2))

        actual = _calculate_response_error_metric(y_true, y_pred, "nrmse")

        self.assertAlmostEqual(actual, expected)

    def test_calculate_response_error_metric_rejects_nlpd(self):
        y_true = np.array([[1.0, 2.0]])
        y_pred = np.array([[1.0, 2.0]])

        with self.assertRaises(ValueError):
            _calculate_response_error_metric(y_true, y_pred, "nlpd")

    def test_check_fields_in_keys_list_raises_for_missing_field(self):
        with self.assertRaises(KeyError):
            _check_fields_in_keys_list(
                ["missing"],
                ["present"],
                "test data set",
            )

    def test_select_state_data_raises_when_multiple_states_and_none_requested(self):
        class FakeDataCollection:
            state_names = ["state_a", "state_b"]

            def __getitem__(self, key):
                return key

        with self.assertRaises(ValueError):
            _select_state_data(None, FakeDataCollection())

    def test_parse_evaluation_info_rejects_invalid_type(self):
        with self.assertRaises(TypeError):
            _parse_evaluation_info(object(), None)

    def test_get_n_points_uses_field_length_when_no_interpolation_locations(self):
        n_points = _get_n_points(
            interpolation_locations=None,
            training_data_list=[{"response": np.arange(7)}],
            field="response",
        )

        self.assertEqual(n_points, 7)

    def test_process_interpolation_locations_returns_none_without_interpolation_field(self):
        value = _process_interpolation_locations(
            output_history=None,
            interpolation_locations=10,
            interpolation_field=None,
        )

        self.assertIsNone(value)

    def test_process_interpolation_locations_rejects_non_array_like(self):
        class BadInterpolationLocations:
            pass

        with self.assertRaises(ValueError):
            _process_interpolation_locations(
                output_history=None,
                interpolation_locations=BadInterpolationLocations(),
                interpolation_field="x",
            )

    def test_ensure_2d_array_converts_non_numpy_input(self):
        result = _ensure_2d_array([1.0, 2.0, 3.0])

        expected = np.array([[1.0], [2.0], [3.0]])
        self.assert_close_arrays(result, expected)

    def test_convert_param_array_to_dict_returns_dict_input_unchanged(self):
        params = OrderedDict()
        params["a"] = np.array([1.0])
        params["b"] = np.array([2.0])

        result = _convert_param_array_to_dict(params, ["a", "b"])

        self.assertIs(result, params)

    def test_prepare_metric_arrays_rejects_incompatible_shapes(self):
        reference = np.array([[1.0, 2.0], [3.0, 4.0]])
        prediction = np.array([1.0, 2.0, 3.0])

        with self.assertRaises(RuntimeError):
            _prepare_metric_arrays(reference, prediction)

    def test_normalized_rmse_falls_back_to_rmse_for_zero_reference_norm(self):
        reference = np.zeros((2, 2))
        prediction = np.ones((2, 2))

        expected = _root_mean_squared_error(reference, prediction)
        actual = _normalized_root_mean_squared_error(reference, prediction)

        self.assertAlmostEqual(actual, expected)

    def test_prepare_metric_arrays_rejects_same_size_different_shape(self):
        reference = np.array([[1.0, 2.0], [3.0, 4.0]])
        prediction = np.array([1.0, 2.0, 3.0, 4.0])

        with self.assertRaises(RuntimeError):
            _prepare_metric_arrays(reference, prediction)

    def test_prepare_metric_arrays_accepts_matching_shapes(self):
        reference = np.array([[1.0, 2.0], [3.0, 4.0]])
        prediction = np.array([[1.0, 2.0], [3.0, 4.0]])

        ref_out, pred_out = _prepare_metric_arrays(reference, prediction)

        self.assert_close_arrays(ref_out, reference)
        self.assert_close_arrays(pred_out, prediction)

    def test_calculate_response_error_metric_max_error(self):
        y_true = np.array([[1.0, 2.0], [3.0, 4.0]])
        y_pred = np.array([[1.0, -1.0], [10.0, 4.0]])

        actual = _calculate_response_error_metric(y_true, y_pred, "max_error")

        self.assertAlmostEqual(actual, 7.0)

    def test_calculate_response_error_metric_r2_and_score(self):
        y_true = np.array([[1.0, 2.0], [3.0, 4.0]])
        y_pred = y_true.copy()

        self.assertAlmostEqual(
            _calculate_response_error_metric(y_true, y_pred, "r2"),
            1.0,
        )
        self.assertAlmostEqual(
            _calculate_response_error_metric(y_true, y_pred, "score"),
            1.0,
        )

    def test_calculate_performance_metrics_uses_none_score_for_one_sample(self):
        class FakeRegressor:
            def score(self, input_values, output_values):
                raise RuntimeError("score should not be called for one sample")

            def predict(self, input_values, return_std=False):
                if return_std:
                    raise RuntimeError("No predictive standard deviation")
                return np.zeros((1, 2))

        metrics = _calculate_performance_metrics(
            FakeRegressor(),
            param=np.array([[0.0]]),
            data=np.array([[1.0, 2.0]]),
        )

        self.assertIsNone(metrics[0])
        self.assertTrue(np.isnan(metrics[1]))
        self.assertAlmostEqual(metrics[2], np.sqrt((1.0**2 + 2.0**2) / 2.0))

    def test_apply_regressor_metric_non_modal_regressor_path(self):
        class FakeRegressor:
            pass

        def metric_func(regressor, input_values, y_true):
            return 12.5

        value = _apply_regressor_metric(
            FakeRegressor(),
            input_values=np.array([[0.0]]),
            evals=np.array([[1.0]]),
            metric_func=metric_func,
        )

        self.assertEqual(value, 12.5)

    def test_modal_regressor_fit_rejects_inconsistent_input_size(self):
        regressor = _modal_regressor(
            regressor_type="RBF",
            n_inputs=2,
            regressor_kwargs={"neighbors": 2},
        )

        with self.assertRaises(ValueError):
            regressor.fit(
                input_values=np.zeros((3, 1)),
                mode_values=np.zeros((3, 1)),
            )

    def test_decompose_with_pca_logs_non_real_option(self):
        rng = np.random.default_rng(123)
        data = rng.normal(size=(20, 5))

        transformed, pca = _decompose_with_pca(
            data,
            var_tol="mle",
            logger_on=True,
        )

        self.assertEqual(transformed.shape[0], data.shape[0])
        self.assertTrue(hasattr(pca, "n_components_"))

    def test_tune_data_decomposition_logs_max_modes_reached(self):
        rng = np.random.default_rng(123)
        source_data = rng.normal(size=(8, 20))

        data_scaler, decomposer, scaled_latent_data, latent_scaler = (
            _tune_data_decomposition(
                source_data,
                make_log_scale=False,
                reconstruction_error_tol=1.0e-300,
                max_modes=1,
                logger_on=False,
            )
        )

        self.assertIsNotNone(data_scaler)
        self.assertIsNotNone(decomposer)
        self.assertIsNotNone(latent_scaler)
        self.assertEqual(scaled_latent_data.shape[0], source_data.shape[0])

    def test_record_variance_behaviors_saves_plot(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        class FakeDecomposer:
            explained_variance_ratio_ = np.array([0.8, 0.15, 0.05])

        with patch("matplotlib.pyplot.savefig") as savefig_mock:
            _record_variance_behaviors(
                FakeDecomposer(),
                filename_base="variance_test",
                field_name="response",
            )

        savefig_mock.assert_called_once()
        self.assertEqual(
            savefig_mock.call_args[0][0],
            "variance_test_response_pca_variance.png",
        )

        plt.close("all")

    def test_rbf_interpolator_rejects_non_2d_input_values(self):
        regressor = _RBFInterpolatorRegressor()

        with self.assertRaises(ValueError):
            regressor.fit(
                input_values=np.array([0.0, 1.0, 2.0]),
                output_values=np.array([0.0, 1.0, 2.0]),
            )

    def test_rbf_interpolator_accepts_neighbors_none(self):
        x = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ])
        y = x[:, 0] + x[:, 1]

        regressor = _RBFInterpolatorRegressor(
            neighbors=None,
            kernel="linear",
            degree=0,
        )

        regressor.fit(x, y)

        self.assertIsNone(regressor._effective_neighbors)

    def test_rbf_interpolator_rejects_nonpositive_neighbors(self):
        x = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
        ])
        y = np.array([0.0, 1.0])

        regressor = _RBFInterpolatorRegressor(neighbors=0)

        with self.assertRaises(ValueError):
            regressor.fit(x, y)

    def test_rbf_interpolator_predict_before_fit_raises(self):
        regressor = _RBFInterpolatorRegressor()

        with self.assertRaises(RuntimeError):
            regressor.predict(np.array([[0.0, 0.0]]))

    def test_rbf_interpolator_score_reshapes_column_output(self):
        x = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ])
        y = x[:, 0] + x[:, 1]

        regressor = _RBFInterpolatorRegressor(
            neighbors=4,
            kernel="linear",
            degree=0,
        )
        regressor.fit(x, y)

        score = regressor.score(x, y.reshape(-1, 1))

        self.assertGreater(score, 0.99)

    def test_format_ax_set_wraps_single_axes(self):
        sur_gen = SurrogateGenerator(
            {"input": {}, "output": {}},
            training_fraction=0.8,
        )

        ax = object()
        formatted = sur_gen._format_ax_set(1, ax)

        self.assertEqual(formatted, [ax])

    def test_plot_set_without_prediction_locations(self):
        class FakeAxes:
            def __init__(self):
                self.plot_calls = []
                self.xlabel = None
                self.ylabel = None
                self.title = None
                self.legend_called = False

            def plot(self, *args, **kwargs):
                self.plot_calls.append((args, kwargs))

            def set_xlabel(self, value):
                self.xlabel = value

            def set_ylabel(self, value):
                self.ylabel = value

            def set_title(self, value):
                self.title = value

            def legend(self):
                self.legend_called = True

        class FakeSurrogate:
            prediction_locations = None
            independent_field = None

        sur_gen = SurrogateGenerator(
            {"input": {}, "output": {}},
            training_fraction=0.8,
        )

        axes = FakeAxes()

        source_data = {
            "response": np.array([[1.0, 2.0, 3.0]])
        }
        surrogate_prediction = {
            "response": np.array([[1.1, 1.9, 3.2]])
        }

        sur_gen._plot_set(
            FakeSurrogate(),
            source_data,
            surrogate_prediction,
            axes,
            "response",
            0,
        )

        self.assertEqual(len(axes.plot_calls), 2)
        self.assertIsNone(axes.xlabel)
        self.assertEqual(axes.ylabel, "response")
        self.assertIn("response eval index0", axes.title)
        self.assertTrue(axes.legend_called)

    def test_get_worst_recreations_returns_requested_number(self):
        sur_gen = SurrogateGenerator(
            {"input": {}, "output": {}},
            training_fraction=0.8,
        )

        source_data = {
            "response": np.array([
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
            ])
        }

        surrogate_prediction = {
            "response": np.array([
                [1.0, 2.0],
                [10.0, 10.0],
                [5.1, 6.1],
            ])
        }

        worst = sur_gen._get_worst_recreations(
            source_data,
            n_worst=2,
            n_eval=3,
            sur_predict=surrogate_prediction,
        )

        self.assertEqual(len(worst), 2)

    def test_parameter_scaler_set_inverse_transform_as_array(self):
        params = OrderedDict()
        params["a"] = np.array([0.0, 1.0, 2.0])
        params["b"] = np.array([-1.0, 0.0, 1.0])

        scaler_set = _make_parameter_scaler_set(params, fields_to_log_scale=["a", "b"])

        scaled = scaler_set.transform_as_array(params.copy())
        recovered = scaler_set.inverse_transform_as_array(scaled.copy())

        expected = np.column_stack((params["a"], params["b"]))
        self.assert_close_arrays(recovered, expected)

    def test_calculate_nlpd_reshapes_mean_prediction(self):
        class FakeGPR:
            def predict(self, input_values, return_std=False):
                mu = np.array([1.0, 2.0, 3.0, 4.0])
                std = np.array([1.0, 2.0])
                return mu, std

        y_true = np.array([
            [1.0, 2.0],
            [3.0, 4.0],
        ])

        value = _calculate_nlpd(
            FakeGPR(),
            input_values=np.zeros((2, 1)),
            y_true=y_true,
        )

        self.assertTrue(np.isfinite(value))

    def test_calculate_nlpd_reshapes_std_prediction(self):
        class FakeGPR:
            def predict(self, input_values, return_std=False):
                mu = np.array([
                    [1.0, 2.0],
                    [3.0, 4.0],
                ])
                std = np.array([1.0, 1.0, 2.0, 2.0])
                return mu, std

        y_true = np.array([
            [1.0, 2.0],
            [3.0, 4.0],
        ])

        value = _calculate_nlpd(
            FakeGPR(),
            input_values=np.zeros((2, 1)),
            y_true=y_true,
        )

        self.assertTrue(np.isfinite(value))

    def test_calculate_nlpd_rejects_incompatible_mean_shape(self):
        class FakeGPR:
            def predict(self, input_values, return_std=False):
                mu = np.array([1.0, 2.0, 3.0])
                std = np.ones((2, 2))
                return mu, std

        y_true = np.ones((2, 2))

        with self.assertRaises(RuntimeError):
            _calculate_nlpd(
                FakeGPR(),
                input_values=np.zeros((2, 1)),
                y_true=y_true,
            )

    def test_calculate_nlpd_rejects_incompatible_std_shape(self):
        class FakeGPR:
            def predict(self, input_values, return_std=False):
                mu = np.ones((2, 2))
                std = np.array([1.0, 2.0, 3.0])
                return mu, std

        y_true = np.ones((2, 2))

        with self.assertRaises(RuntimeError):
            _calculate_nlpd(
                FakeGPR(),
                input_values=np.zeros((2, 1)),
                y_true=y_true,
            )

    def test_prepare_metric_arrays_accepts_column_reference_and_1d_prediction(self):
        reference = np.array([[1.0], [2.0], [3.0]])
        prediction = np.array([1.0, 2.0, 4.0])

        ref_out, pred_out = _prepare_metric_arrays(reference, prediction)

        self.assertEqual(ref_out.shape, (3, 1))
        self.assertEqual(pred_out.shape, (3, 1))
        self.assert_close_arrays(pred_out, np.array([[1.0], [2.0], [4.0]]))

    def test_prepare_metric_arrays_rejects_row_vector_and_1d_vector(self):
        reference = np.array([[1.0, 2.0, 3.0]])
        prediction = np.array([1.0, 2.0, 3.0])

        with self.assertRaises(RuntimeError):
            _prepare_metric_arrays(reference, prediction)


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
            worst_scores = surrogate.scores['test'][field]
            if isinstance(worst_scores, (float, int)):
                self.assertGreaterEqual(worst_scores, 0.99)
            else:
                for idx in range(len(worst_scores)):
                    self.assertGreaterEqual(worst_scores[idx], 0.99)

    def _confirm_alignment_to_function(self, p_low, p_high, show_array, probes, 
                                       err_tol, n_interp, test_function, surrogate, 
                                       scale_factor=1, log_indices=[]):
        test_sets = _make_test_sets_uniform(p_low, p_high, log_indices)
        surrogate.enforce_training_data_parameter_range(False)
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

    def test_surrogate_confirm_error_for_bad_calls(self):
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
        with self.assertRaises(RuntimeError):
            surrogate(-1,0)
        with self.assertRaises(RuntimeError):
            surrogate(0.5,2)
        surrogate.enforce_training_data_parameter_range(False)
        vals = surrogate(-1,0)
        surrogate.enforce_training_data_parameter_range(True)

        self.assertIsInstance(vals, OrderedDict)

        # Wrong number of positional arguments (only one while two are required)
        with self.assertRaises(RuntimeError):
            surrogate(0.1)  # missing second parameter

        # Incomplete keyword dict (missing 'b')
        with self.assertRaises(RuntimeError):
            surrogate(m=0.2)

        # Incorrect keyword dict 'a' != 'b')
        with self.assertRaises(RuntimeError):
            surrogate(m=0.2, a=0.1)

        # Both positional and keyword arguments together – also invalid
        with self.assertRaises(RuntimeError):
            surrogate(0.1, b=0.3)

        #outside of bound (-5,5) for both params
        with self.assertRaises(RuntimeError):
            surrogate(-10, 0)
        with self.assertRaises(RuntimeError):
            surrogate(0, 10)

        with self.assertRaises(RuntimeError):
            surrogate([[0, 10], [1, 0]], batch_evaluate=True)
        #verify it takes in kwargs with lists of param values
        res = surrogate(m=[0.1, 0.8], b=[0.1, 0.8])
        self.assertEqual(res["y"].shape, (2,200))

    def test_set_param_ranges(self):
        def test_function(m, b, n_features=None):
            if n_features == None:
                n_features = np.random.randint(10, 50)
            x = np.linspace(0, 10, n_features)
            y = m * x + b
            return {'x':x, 'y':y}

        n_samples = 20
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
        surrogate.set_parameter_ranges(m=[-10, 10], b=[0, 20])
        self.assert_close_dicts_or_data(surrogate._param_ranges, {"m": [-10, 10], "b":[0, 20]})
        with self.assertRaises(RuntimeError):
            surrogate.set_parameter_ranges(1, 1)
        with self.assertRaises(RuntimeError):
            surrogate.set_parameter_ranges(a=[1, 2], b=[1, 2])
        with self.assertRaises(RuntimeError):
            surrogate.set_parameter_ranges(m=[1, 2, 1], b=[1, 2])
        with self.assertRaises(RuntimeError):
            surrogate.set_parameter_ranges(b=[1, 2])
        with self.assertRaises(TypeError):
            surrogate.set_parameter_ranges(b=["a", 2])
        with self.assertRaises(ValueError):
            surrogate.set_parameter_ranges(b=[2, 1])          

    def test_surrogate_for_line_training_fraction_1(self):
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
                                                     indep_var, test_function, training_fraction=1.0)
        sur_gen.set_surrogate_details("PCA Multiple Regressors", "Gaussian Process", 1.0)
        surrogate = sur_gen.generate('my_surrogate')

        self._confirm_alignment_to_function(p_low, p_high, show_array, probes, err_tol, n_interp, 
                                            test_function, surrogate)
        self._confirm_good_test_scores(surrogate)

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
                predictions = surrogate(np.array(test_set).reshape(1, -1), batch_evaluate=True)
                prediction = predictions[test_field]
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
        prediction_keys = list(predictions.keys())
        for prediction_key in prediction_keys:
            if prediction_key != surrogate._interpolation_field:
                self.assertIn(prediction_key, probes)
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

    def test_surrogate_for_line_with_rbf_regressor(self):
        def test_function(m, b, n_features=None):
            if n_features is None:
                n_features = np.random.randint(10, 50)
            x = np.linspace(0, 10, n_features)
            y = m * x + b
            return {'x': x, 'y': y}

        n_samples = 200
        p_names = ['m', 'b']
        p_low = [0, -1]
        p_high = [1, 1]
        show_array = True
        probes = ['y']
        indep_var = 'x'
        err_tol = 5e-2
        n_interp = 200

        sur_gen = _setup_initial_surrogate_generator(
            n_samples,
            p_names,
            p_low,
            p_high,
            indep_var,
            test_function,
        )

        sur_gen.set_surrogate_details(
            surrogate_type="PCA Multiple Regressors",
            regressor_type="RBF",
            neighbors=25,
        )

        surrogate = sur_gen.generate('my_rbf_surrogate')

        self._confirm_alignment_to_function(
            p_low,
            p_high,
            show_array,
            probes,
            err_tol,
            n_interp,
            test_function,
            surrogate,
        )

        self._confirm_good_test_scores(surrogate)

        for field, modal_regressor in surrogate._regressors.items():
            for mode_regressor in modal_regressor._mode_regressors:
                self.assertIsInstance(mode_regressor, _RBFInterpolatorRegressor)
                self.assertEqual(mode_regressor._effective_neighbors, 25)

    def test_generate_with_none_save_filename_does_not_serialize_surrogate(self):
        def test_function(m, b, n_features=None):
            x = np.linspace(0, 1, 5)
            y = m * x + b
            return {"x": x, "y": y}

        sur_gen = _setup_initial_surrogate_generator(
            n_samples=30,
            p_names=["m", "b"],
            p_low=[0.0, -1.0],
            p_high=[1.0, 1.0],
            indep_var="x",
            test_function=test_function,
            interp_locations=np.linspace(0, 1, 5),
        )

        sur_gen.set_surrogate_details(
            surrogate_type="PCA Multiple Regressors",
            regressor_type="RBF",
            neighbors=5,
        )

        with patch("matcal.core.surrogates.matcal_save") as save_mock:
            surrogate = sur_gen.generate(None)

        self.assertIsNotNone(surrogate)
        save_mock.assert_not_called()

    def test_generate_with_save_filename_serializes_surrogate(self):
        def test_function(m, b, n_features=None):
            x = np.linspace(0, 1, 5)
            y = m * x + b
            return {"x": x, "y": y}

        sur_gen = _setup_initial_surrogate_generator(
            n_samples=30,
            p_names=["m", "b"],
            p_low=[0.0, -1.0],
            p_high=[1.0, 1.0],
            indep_var="x",
            test_function=test_function,
            interp_locations=np.linspace(0, 1, 5),
        )

        sur_gen.set_surrogate_details(
            surrogate_type="PCA Multiple Regressors",
            regressor_type="RBF",
            neighbors=5,
        )

        with patch("matcal.core.surrogates.matcal_save") as save_mock:
            surrogate = sur_gen.generate("saved_test_surrogate")

        self.assertIsNotNone(surrogate)
        save_mock.assert_called_once()
        self.assertEqual(save_mock.call_args[0][0], "saved_test_surrogate.joblib")

    def test_generate_with_none_save_filename_and_plot_worst_raises(self):
        def test_function(m, b, n_features=None):
            x = np.linspace(0, 1, 5)
            y = m * x + b
            return {"x": x, "y": y}

        sur_gen = _setup_initial_surrogate_generator(
            n_samples=30,
            p_names=["m", "b"],
            p_low=[0.0, -1.0],
            p_high=[1.0, 1.0],
            indep_var="x",
            test_function=test_function,
            interp_locations=np.linspace(0, 1, 5),
        )

        with self.assertRaises(ValueError):
            sur_gen.generate(None, plot_n_worst=1)

    def test_set_model_and_state_sets_values(self):
        sur_gen = SurrogateGenerator(
            {"input": {}, "output": {}},
            training_fraction=0.8,
        )

        sur_gen.set_model_and_state(model_name="model_a", state="state_a")

        self.assertEqual(sur_gen._model_name, "model_a")
        self.assertEqual(sur_gen._state, "state_a")

    def test_set_surrogate_details_updates_test_eval_info_when_provided(self):
        sur_gen = SurrogateGenerator(
            {"input": {}, "output": {}},
            training_fraction=0.8,
        )

        test_eval_info = object()

        sur_gen.set_surrogate_details(
            training_fraction=1.0,
            test_eval_info=test_eval_info,
        )

        self.assertIs(sur_gen._test_eval_info, test_eval_info)

    def test_set_fields_to_log_scale_sets_fields(self):
        sur_gen = SurrogateGenerator(
            {"input": {}, "output": {}},
            training_fraction=0.8,
        )

        sur_gen.set_fields_to_log_scale("a", "b")

        self.assertEqual(sur_gen._fields_to_log_scale, ("a", "b"))

    def test_training_fraction_one_without_test_eval_info_raises(self):
        with self.assertRaises(ValueError):
            SurrogateGenerator(
                {"input": {}, "output": {}},
                training_fraction=1.0,
            )

    def test_generated_surrogate_public_properties_are_covered(self):
        def test_function(m, b, n_features=None):
            x = np.linspace(0.0, 1.0, 5)
            y = m * x + b
            return {"x": x, "y": y}

        sur_gen = _setup_initial_surrogate_generator(
            n_samples=40,
            p_names=["m", "b"],
            p_low=[0.0, -1.0],
            p_high=[1.0, 1.0],
            indep_var="x",
            test_function=test_function,
            interp_locations=np.linspace(0.0, 1.0, 5),
        )

        sur_gen.set_surrogate_details(
            surrogate_type="PCA Multiple Regressors",
            regressor_type="RBF",
            neighbors=5,
            kernel="linear",
            degree=0,
        )

        surrogate = sur_gen.generate(None)

        self.assertEqual(surrogate.parameter_order, ["m", "b"])
        self.assertEqual(surrogate.independent_field, "x")
        self.assert_close_arrays(
            surrogate.prediction_locations,
            np.linspace(0.0, 1.0, 5),
        )
        self.assertIn("test", surrogate.max_errors)
        self.assertIn("test", surrogate.rmse_errors)

    def test_generate_monolithic_pca_surrogate(self):
        def test_function(m, b, n_features=None):
            x = np.linspace(0.0, 1.0, 5)
            y = m * x + b
            return {"x": x, "y": y}

        sur_gen = _setup_initial_surrogate_generator(
            n_samples=40,
            p_names=["m", "b"],
            p_low=[0.0, -1.0],
            p_high=[1.0, 1.0],
            indep_var="x",
            test_function=test_function,
            interp_locations=np.linspace(0.0, 1.0, 5),
        )

        sur_gen.set_surrogate_details(
            surrogate_type="PCA Monolithic Regressor",
            regressor_type="RBF",
            neighbors=5,
            kernel="linear",
            degree=0,
        )

        surrogate = sur_gen.generate(None)

        self.assertIsNotNone(surrogate)
        self.assertIsInstance(surrogate, MatCalMonolithicPCASurrogate)

    def test_generate_with_plot_n_worst_covers_plot_path(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        def test_function(m, b, n_features=None):
            x = np.linspace(0.0, 1.0, 5)
            y = m * x + b
            return {"x": x, "y": y}

        sur_gen = _setup_initial_surrogate_generator(
            n_samples=30,
            p_names=["m", "b"],
            p_low=[0.0, -1.0],
            p_high=[1.0, 1.0],
            indep_var="x",
            test_function=test_function,
            interp_locations=np.linspace(0.0, 1.0, 5),
        )

        sur_gen.set_surrogate_details(
            surrogate_type="PCA Multiple Regressors",
            regressor_type="RBF",
            neighbors=5,
            kernel="linear",
            degree=0,
        )

        with patch("matcal.core.surrogates.matcal_save"):
            with patch("matplotlib.pyplot.savefig") as savefig_mock:
                surrogate = sur_gen.generate(
                    "plot_worst_surrogate",
                    plot_n_worst=2,
                )

        self.assertIsNotNone(surrogate)
        savefig_mock.assert_called_once()

        plt.close("all")

    def test_generate_with_none_save_filename_does_not_serialize_surrogate(self):
        def test_function(m, b, n_features=None):
            x = np.linspace(0.0, 1.0, 5)
            y = m * x + b
            return {"x": x, "y": y}

        sur_gen = _setup_initial_surrogate_generator(
            n_samples=30,
            p_names=["m", "b"],
            p_low=[0.0, -1.0],
            p_high=[1.0, 1.0],
            indep_var="x",
            test_function=test_function,
            interp_locations=np.linspace(0.0, 1.0, 5),
        )

        sur_gen.set_surrogate_details(
            surrogate_type="PCA Multiple Regressors",
            regressor_type="RBF",
            neighbors=5,
            kernel="linear",
            degree=0,
        )

        with patch("matcal.core.surrogates.matcal_save") as save_mock:
            surrogate = sur_gen.generate(None)

        self.assertIsNotNone(surrogate)
        save_mock.assert_not_called()

    def test_generate_with_save_filename_serializes_surrogate(self):
        def test_function(m, b, n_features=None):
            x = np.linspace(0.0, 1.0, 5)
            y = m * x + b
            return {"x": x, "y": y}

        sur_gen = _setup_initial_surrogate_generator(
            n_samples=30,
            p_names=["m", "b"],
            p_low=[0.0, -1.0],
            p_high=[1.0, 1.0],
            indep_var="x",
            test_function=test_function,
            interp_locations=np.linspace(0.0, 1.0, 5),
        )

        sur_gen.set_surrogate_details(
            surrogate_type="PCA Multiple Regressors",
            regressor_type="RBF",
            neighbors=5,
            kernel="linear",
            degree=0,
        )

        with patch("matcal.core.surrogates.matcal_save") as save_mock:
            surrogate = sur_gen.generate("saved_test_surrogate")

        self.assertIsNotNone(surrogate)
        save_mock.assert_called_once()
        self.assertEqual(save_mock.call_args[0][0], "saved_test_surrogate.joblib")

    def test_generate_with_none_save_filename_and_plot_worst_raises(self):
        sur_gen = SurrogateGenerator(
            {"input": {}, "output": {}},
            training_fraction=0.8,
        )

        with self.assertRaises(ValueError):
            sur_gen.generate(None, plot_n_worst=1)


class TestProcessSurrogateArgsCall(MatcalUnitTest):

    def setUp(self):
        super().setUp(__file__)

    def test_batch_evaluate_array(self):
        """batch_evaluate=True should return the input array unchanged (as float)."""
        arr = np.array([[1, 2], [3, 4]], dtype=float)
        out = _process_surrogate_args_call(['a', 'b'], arr, batch_evaluate=True)
        np.testing.assert_array_equal(out, arr)

    def test_dict_argument_respects_order(self):
        """When a single dict is passed, the resulting array must follow `param_names` order."""
        param_names = ['alpha', 'beta']
        data = {'beta': [10, 20], 'alpha': [1, 2]}  # order intentionally swapped
        out = _process_surrogate_args_call(param_names, data)
        expected = np.array([[1, 10],
                             [2, 20]], dtype=float)
        np.testing.assert_array_equal(out, expected)

    def test_positional_arguments_match_param_names(self):
        """Exact positional arguments equal to the number of parameters should be returned."""
        param_names = ['x', 'y', 'z']
        out = _process_surrogate_args_call(param_names, 1, 2, 3)
        expected = np.array([1, 2, 3], dtype=float)
        np.testing.assert_array_equal(out, expected)

    def test_keyword_arguments_match_param_names(self):
        """All parameters supplied as keywords should be converted to the correct array."""
        param_names = ['x', 'y', 'z']
        out = _process_surrogate_args_call(param_names, x=1, y=2, z=3)
        expected = np.array([1, 2, 3], dtype=float)
        np.testing.assert_array_equal(out, expected)

    def test_invalid_mixed_positional_and_kwargs(self):
        """Providing a mix of positional args and kwargs should raise RuntimeError."""
        param_names = ['x', 'y', 'z']
        with self.assertRaises(RuntimeError):
            _process_surrogate_args_call(param_names, 1, 2, y=3)

    def test_invalid_wrong_number_of_positional(self):
        """Supplying fewer positional arguments than required should raise RuntimeError."""
        param_names = ['a', 'b', 'c']
        with self.assertRaises(RuntimeError):
            _process_surrogate_args_call(param_names, 1, 2)  # only two values provided

    def test_dict_missing_parameter(self):
        """A dict missing any of the required parameter names should raise RuntimeError."""
        param_names = ['a', 'b']
        data = {'a': [1, 2]}  # 'b' is missing
        with self.assertRaises(RuntimeError):
            _process_surrogate_args_call(param_names, data)