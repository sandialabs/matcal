from matcal.core.constants import BATCH_RESTART_FILENAME
import numpy as np

from matcal.core.calibration_studies import (ScipyMinimizeStudy, ScipyLeastSquaresStudy, 
                                             _AlgorithmOptions, )
from matcal.core.data import convert_dictionary_to_data
from matcal.core.models import PythonModel
from matcal.core.parameters import Parameter, ParameterCollection
from matcal.core.objective import CurveBasedInterpolatedObjective
from matcal.core.tests.MatcalUnitTest import MatcalUnitTest
from matcal.core.tests.unit.test_study_base import StudyBaseUnitTests
import os, gc


def my_function(**param_dict):
    import numpy as np
    x = np.linspace(0, 10, 25)
    y = param_dict['m'] * x
    return {"x": x, "y": y}


def get_goal_data(ref_value):
    res_dict = my_function(m=ref_value)
    res_dict['not_used'] = res_dict['x'] * -5
    return convert_dictionary_to_data(res_dict)


def run_study_for_method(test, method=None, study=ScipyMinimizeStudy, 
                         step_size=1e-6, 
                         initial_guess=100.0,
                         central_finite_diff=False, metric_function=None, 
                         **kwargs):
    slope = Parameter("m", -100., 100., initial_guess)
    parameter_collection = ParameterCollection("one_parameter", slope)

    goal_value = 2.
    curve_data = get_goal_data(goal_value)

    model = PythonModel(my_function)

    calibration = study(parameter_collection, method=method,
                         **kwargs)
    objective = CurveBasedInterpolatedObjective("x", "y")
    if metric_function != None:
        objective.set_metric_function(metric_function)
    calibration.add_evaluation_set(model, objective, curve_data)
    calibration.set_core_limit(6)
    calibration.set_step_size(step_size)
    calibration.use_three_point_finite_difference(central_finite_diff)
    results = calibration.launch()
    test.assertAlmostEqual(results.outcome['best:m'], goal_value, delta=goal_value * test.error_tol)

def run_serial_study_for_method(test, method=None, study=ScipyMinimizeStudy, 
                         step_size=1e-6, 
                         initial_guess=100.0,
                         central_finite_diff=False, metric_function=None, 
                         **kwargs):
    slope = Parameter("m", -100., 100., initial_guess)
    parameter_collection = ParameterCollection("one_parameter", slope)

    goal_value = 2.
    curve_data = get_goal_data(goal_value)

    model = PythonModel(my_function)

    calibration = study(parameter_collection, method=method,
                         **kwargs)
    objective = CurveBasedInterpolatedObjective("x", "y")
    if metric_function != None:
        objective.set_metric_function(metric_function)
    calibration.add_evaluation_set(model, objective, curve_data)
    calibration.set_core_limit(6)
    calibration.set_step_size(step_size)
    calibration.use_three_point_finite_difference(central_finite_diff)
    calibration.run_in_serial()
    results = calibration.launch()
    test.assertAlmostEqual(results.outcome['best:m'], goal_value, delta=goal_value * test.error_tol)

def run_serial_study_for_method_get_history(test, method=None, study=ScipyMinimizeStudy, 
                         step_size=1e-6, 
                         initial_guess=100.0,
                         central_finite_diff=False, metric_function=None, 
                         **kwargs):
    slope = Parameter("m", -100., 100., initial_guess)
    parameter_collection = ParameterCollection("one_parameter", slope)

    goal_value = 2.
    curve_data = get_goal_data(goal_value)

    model = PythonModel(my_function)

    calibration = study(parameter_collection, method=method,
                         **kwargs)
    objective = CurveBasedInterpolatedObjective("x", "y")
    if metric_function != None:
        objective.set_metric_function(metric_function)
    calibration.add_evaluation_set(model, objective, curve_data)
    calibration.set_core_limit(6)
    calibration.set_step_size(step_size)
    calibration.use_three_point_finite_difference(central_finite_diff)
    calibration.run_in_serial()
    calibration.set_results_storage_options(weighted_conditioned=True)
    results = calibration.launch()
    test.assertAlmostEqual(results.outcome['best:m'], goal_value, delta=goal_value * test.error_tol)
    goal_message_start = 'Success'
    test.assertTrue(results.exit_status[:len(goal_message_start)], goal_message_start)
    test.assertAlmostEqual(results.parameter_history['m'][0], 100)
    test.assertAlmostEqual(results.parameter_history['m'][-1], goal_value, delta=goal_value * test.error_tol)

    test.assertEqual(len(results.evaluation_sets), 1)
    goal_name = ":".join([model.name, objective.name])
    test.assertEqual(results.evaluation_sets[0], goal_name)

    test.assertEqual(len(results.evaluation_sets), len(results.objective_history))
    obj_info = results.objective_history[goal_name]
    
    results.objective_history[goal_name]
    
    n_fields = 1
    n_evals = len(results.parameter_history['m'])
    
    test.assertEqual(len(obj_info.residuals[0][curve_data.state][0].field_names), n_fields)
    test.assertEqual(len(obj_info.residuals), n_evals)
    
    #confrim that residual history and parameter history align 
    for eval_idx, m in enumerate(results.parameter_history['m']):
        test_resid = my_function(m=m)['y'] - my_function(m=goal_value)['y']
        test.assert_close_arrays(test_resid, obj_info.residuals[eval_idx][curve_data.state][0], 
                                 show_on_fail=True)
    
    test.assertEqual(len(obj_info.weighted_conditioned_residuals[0][curve_data.state][0].field_names), n_fields)
    test.assertEqual(len(obj_info.weighted_conditioned_residuals), n_evals)
    
    test.assertEqual(len(obj_info.objectives), n_evals)

    test.assertEqual(len(results.total_objective_history), n_evals)
    
    study_qoi_info = results.qoi_history[goal_name]
    study_simulation_hist = results.simulation_history[model.name]
    n_exp_data_fields = 2
    exp_repeats = 1
    test.assertEqual(len(study_qoi_info.experiment_data[curve_data.state][0].field_names), n_exp_data_fields)
    test.assertEqual(len(study_qoi_info.experiment_data[curve_data.state]), exp_repeats)
    test.assert_close_arrays(study_qoi_info.experiment_data[curve_data.state][0]['x'], curve_data['x'])
    test.assert_close_arrays(study_qoi_info.experiment_data[curve_data.state][0]['y'], curve_data['y'])

    n_qois = 3
    test.assertEqual(len(study_qoi_info.experiment_qois[curve_data.state][0].field_names), n_qois)
    test.assertEqual(len(study_qoi_info.experiment_qois[curve_data.state]), exp_repeats)
    test.assertIn('y', study_qoi_info.experiment_qois[curve_data.state][0].field_names)
    test.assertIn('x', study_qoi_info.experiment_qois[curve_data.state][0].field_names)
    test.assert_close_arrays(study_qoi_info.experiment_qois[curve_data.state][0]['y'], curve_data['y'])
    test.assert_close_arrays(study_qoi_info.experiment_qois[curve_data.state][0]['x'], curve_data['x'])

    test.assertEqual(len(study_qoi_info.experiment_weighted_conditioned_qois[curve_data.state][0].field_names), n_qois)
    test.assertEqual(len(study_qoi_info.experiment_weighted_conditioned_qois[curve_data.state]), exp_repeats)
    test.assertIn('y', study_qoi_info.experiment_weighted_conditioned_qois[curve_data.state][0].field_names)
    test.assertIn('x', study_qoi_info.experiment_weighted_conditioned_qois[curve_data.state][0].field_names)
    test.assert_close_arrays(study_qoi_info.experiment_weighted_conditioned_qois[curve_data.state][0]['y'], curve_data['y']/20, show_on_fail=True)
    test.assert_close_arrays(study_qoi_info.experiment_weighted_conditioned_qois[curve_data.state][0]['x'], curve_data['x']/10, show_on_fail=True)

    for field_name in ['x', 'y']:
        test.assertEqual(len(study_simulation_hist[curve_data.state]), n_evals)
        test.assertTrue(field_name in study_simulation_hist[curve_data.state][0].field_names)

        test.assertEqual(len(study_qoi_info.simulation_qois), n_evals)
        test.assertEqual(len(study_qoi_info.simulation_qois[0][curve_data.state]), exp_repeats)
        test.assertTrue(field_name in study_qoi_info.simulation_qois[0][curve_data.state][0].field_names)

        test.assertEqual(len(study_qoi_info.simulation_weighted_conditioned_qois), n_evals)
        test.assertEqual(len(study_qoi_info.simulation_weighted_conditioned_qois[0][curve_data.state]), exp_repeats)
        test.assertTrue(field_name in study_qoi_info.simulation_weighted_conditioned_qois[0][curve_data.state][0].field_names)

    
    
class TestScipyMinimizeStudy(StudyBaseUnitTests.CommonTests):
    _study_class = ScipyMinimizeStudy
    def setUp(self):
        super().setUp(__file__)

    def test_init_method(self):
        study = self._study_class(self.parameter_collection)
        self.assertEqual(study._method, 'l-bfgs-b')
        study = self._study_class(self.parameter_collection, method='Nelder-mEad')
        self.assertEqual(study._method, 'nelder-mead')

        with self.assertRaises(TypeError):
            study = self._study_class(self.parameter_collection, method=1)
        with self.assertRaises(ValueError):
            study = self._study_class(self.parameter_collection, method='bad-method')

    def test_bounds_in_kwargs(self):
        with self.assertRaises(ValueError):
            study = self._study_class(self.parameter_collection, method='Nelder-mEad', 
                                      bounds=())

    def test_hess_error_in_kwargs(self):
        with self.assertRaises(ValueError):
            study = self._study_class(self.parameter_collection, 
                                      hess='3-point')

    def test_determine_jac(self):
        study = self._study_class(self.parameter_collection, method='Nelder-mEad')
        self.assertEqual(None, study._determine_jacobian_argument())
        with self.assertRaises(ValueError):
            study = self._study_class(self.parameter_collection, method='Nelder-mEad', 
                                      jac='2-point')
        study = self._study_class(self.parameter_collection, method='trust-constr')
        self.assertEqual(study._get_current_gradient_value, 
                         study._determine_jacobian_argument())

        study = self._study_class(self.parameter_collection, method='trust-constr', 
                                  jac='2-point')
        self.assertEqual('2-point', study._determine_jacobian_argument())

    def test_scipy_jac_studie_3_point(self):
        self.error_tol=1e-6
        gc.collect()
        run_serial_study_for_method(self, 'SLSQP', self._study_class, jac='3-point')
        
        
    def test_scipy_jac_studie_2_point(self):
        self.error_tol=1e-6
        gc.collect()
        run_serial_study_for_method(self, 'trust-constr', self._study_class, jac='2-point')

    def test_scipy_studies_new_results(self):
        self.error_tol=1e-5
        run_serial_study_for_method_get_history(self, 'SLSQP', self._study_class, jac='3-point', tol=1e-7)

    def test_update_kwargs_with_hessian_argument(self):
        study = self._study_class(self.parameter_collection, method='trust-constr')
        self.assertTrue('hess' not in 
                         study._update_kwargs_with_hessian_argument(study._kwargs))

        study = self._study_class(self.parameter_collection, method='trust-constr',  
                                  hess='3-point')
        self.assertEqual('3-point', study._update_kwargs_with_hessian_argument(study._kwargs)['hess'])

        study = self._study_class(self.parameter_collection, method='dogleg')
        self.assertEqual(study._get_current_gradient_value, 
                         study._determine_jacobian_argument())
        self.assertEqual(study._get_current_hessian_value, 
                         study._update_kwargs_with_hessian_argument(study._kwargs)['hess'])
    
        study = self._study_class(self.parameter_collection, method='slsqp')
        self.assertTrue('hess' not in 
                         study._update_kwargs_with_hessian_argument(study._kwargs))

        with self.assertRaises(ValueError):
            study = self._study_class(self.parameter_collection, method='slsqp',
                                       hess='3-point')

    def test_use_three_point_finite_difference(self):
        study = self._study_class(self.parameter_collection)
        self.assertEqual(study._three_point_finite_difference, False)
        study.use_three_point_finite_difference()
        self.assertEqual(study._three_point_finite_difference, True)
        study.use_three_point_finite_difference(False)
        self.assertEqual(study._three_point_finite_difference, False)
        with self.assertRaises(TypeError):
            study.use_three_point_finite_difference("a")

    def test_restart(self):
        study = self._study_class(self.parameter_collection)
        with self.assertRaises(NotImplementedError):
            study.restart()      

    def test_set_step_size(self):
        study = self._study_class(self.parameter_collection)
        self.assertEqual(study._step_size, 5e-5)
        with self.assertRaises(ValueError):
            study.set_step_size(-1)
        study.set_step_size(1e-6)
        self.assertEqual(study._step_size, 1e-6)

    def test_prepare_finite_difference_two_point(self):
        study = self._study_class(self.parameter_collection)
        param_set = list(self.parameter_collection.get_current_value_dict().values())
        finite_diff, finite_diff_pts = study._prepare_finite_difference(param_set)
        self.assertEqual(finite_diff._relative_step_size, 5e-5)
        self.assertEqual(len(finite_diff_pts), 2)
        study = self._study_class(self.parameter_collection)
        study.set_step_size(1e-6)
        study.use_three_point_finite_difference()
        finite_diff, finite_diff_pts = study._prepare_finite_difference(param_set)
        self.assertEqual(finite_diff._relative_step_size, 1e-6)
        self.assertEqual(len(finite_diff_pts), 3)

    def test_format_parameters(self):
        study =   self._study_class(self.parameter_collection)

        formatted_params = study._format_parameters([1])   
        from collections import OrderedDict
        self.assertEqual(OrderedDict([('a',1)]), formatted_params)
    
    def test_needs_finite_difference_gradients_default(self):
        study =   self._study_class(self.parameter_collection)
        jac = study._determine_jacobian_argument()
        self.assertTrue(study._needs_finite_difference_gradient)
    
    def test_needs_finite_difference_gradients(self):
        study = self._study_class(self.parameter_collection, method='nelder-mead')
        jac = study._determine_jacobian_argument()
        self.assertFalse(study._needs_finite_difference_gradient)
        study = self._study_class(self.parameter_collection, jac='2-point')
        jac = study._determine_jacobian_argument()
        self.assertFalse(study._needs_finite_difference_gradient)

    def test_needs_finite_difference_hessians(self):
        study =   self._study_class(self.parameter_collection)
        kwargs = study._update_kwargs_with_hessian_argument(study._kwargs)
        self.assertFalse(study._needs_finite_difference_hessian)

        study =   self._study_class(self.parameter_collection, method='dogleg')
        kwargs = study._update_kwargs_with_hessian_argument(study._kwargs)
        self.assertTrue(study._needs_finite_difference_hessian)

        study =   self._study_class(self.parameter_collection, method='trust-constr')
        kwargs = study._update_kwargs_with_hessian_argument(study._kwargs)
        self.assertFalse(study._needs_finite_difference_hessian)

        study =   self._study_class(self.parameter_collection, method='trust-constr', 
                                    hess='3-point')
        kwargs = study._update_kwargs_with_hessian_argument(study._kwargs)
        self.assertFalse(study._needs_finite_difference_hessian)

    def test_matcal_evaluate_parameter_sets_batch_and_get_cur_grad_val(self):
        slope = Parameter("m", -100., 100., 2)
        parameter_collection = ParameterCollection("one_parameter", slope)
        curve_data = get_goal_data(2)
        model = PythonModel(my_function)
        study = self._study_class(parameter_collection)
        x =  np.array([2.0])
        self.assertEqual(study._get_current_gradient_value(x), None)
        
        objective = CurveBasedInterpolatedObjective("x", "y")
        study.add_evaluation_set(model, objective, curve_data)
        study._determine_jacobian_argument()
        study._initialize_study_and_batch_evaluator()
        res = study._matcal_evaluate_parameter_sets_batch(x)
        self.assertEqual(res,0)
        self.assert_close_arrays(study._get_current_gradient_value(x), 0, 
                                 atol=1e-5)
        
    def test_matcal_evaluate_parameter_sets_batch_and_get_cur_hess_val(self):
        slope = Parameter("m", -100., 100., 2)
        parameter_collection = ParameterCollection("one_parameter", slope)
        curve_data = get_goal_data(2)
        model = PythonModel(my_function)
        study = self._study_class(parameter_collection, method='dogleg')
        x =  np.array([2.0])
        self.assertEqual(study._get_current_hessian_value(x), None)
        
        objective = CurveBasedInterpolatedObjective("x", "y")
        study.add_evaluation_set(model, objective, curve_data)
        study._determine_jacobian_argument()
        study._update_kwargs_with_hessian_argument(study._kwargs)
        study._initialize_study_and_batch_evaluator()
        res = study._matcal_evaluate_parameter_sets_batch(x)
        self.assertEqual(res,0)
        self.assert_close_arrays(study._get_current_gradient_value(x), 0, 
                                 atol=1e-5)
        self.assertIsInstance(study._get_current_hessian_value(x)[0,0],float)

    def test_get_bounds(self):
        study = self._study_class(self.parameter_collection)
        bounds = study._get_bounds()
        self.assertEqual([[0.0, 10.0]], bounds)
        study = self._study_class(self.parameter_collection, method='bfgs')
        bounds = study._get_bounds()
        self.assertEqual(None, bounds)



class TestScipyLeastSquaresStudy(StudyBaseUnitTests.CommonTests):
    _study_class = ScipyLeastSquaresStudy
    
    def setUp(self):
        super().setUp(__file__)
    
    
    def test_init_method(self):
        study = self._study_class(self.parameter_collection)
        self.assertEqual(study._method, 'trf')
        study = self._study_class(self.parameter_collection, method='LM')
        self.assertEqual(study._method, 'lm')

        with self.assertRaises(TypeError):
            study = self._study_class(self.parameter_collection, method=1)
        with self.assertRaises(ValueError):
            study = self._study_class(self.parameter_collection, method='bad-method')
        
    def test_scipy_jac_studies_2_point(self):
        self.error_tol=1e-6
        run_study_for_method(self, 'LM', self._study_class, jac='2-point')

    def test_scipy_jac_studies_3_point(self):
        self.error_tol=1e-6
        run_study_for_method(self, 'trf', self._study_class, jac='3-point')

    def test_determine_jac(self):
        study = self._study_class(self.parameter_collection, method='trf')
        self.assertEqual(study._get_current_gradient_value, 
                         study._determine_jacobian_argument())
        study = self._study_class(self.parameter_collection, method='trf', 
                                  jac='3-point')

        self.assertEqual('3-point', 
                         study._determine_jacobian_argument())
    
    def test_update_kwargs_with_hessian_argument(self):
        study = self._study_class(self.parameter_collection)
        self.assertTrue('hess' not in
                         study._update_kwargs_with_hessian_argument(study._kwargs))
        with self.assertRaises(ValueError):
            study = self._study_class(self.parameter_collection, hess='3-point')

    def test_needs_finite_difference_gradients(self):
        study = self._study_class(self.parameter_collection, jac='2-point')
        jac = study._determine_jacobian_argument()
        self.assertFalse(study._needs_finite_difference_gradient)

    def test_needs_finite_difference_hessians(self):
        study =   self._study_class(self.parameter_collection)
        kwargs = study._update_kwargs_with_hessian_argument(study._kwargs)
        self.assertFalse(study._needs_finite_difference_hessian)

    def test_matcal_evaluate_parameter_sets_batch_and_get_cur_grad_val(self):
        slope = Parameter("m", -100., 100., 2)
        parameter_collection = ParameterCollection("one_parameter", slope)
        curve_data = get_goal_data(2)
        model = PythonModel(my_function)
        study = self._study_class(parameter_collection)
        x =  np.array([2.0])
        self.assertEqual(study._get_current_gradient_value(x), None)
        objective = CurveBasedInterpolatedObjective("x", "y")
        study.add_evaluation_set(model, objective, curve_data)
        study.use_three_point_finite_difference()
        study._determine_jacobian_argument()
        study._initialize_study_and_batch_evaluator()
        res = study._matcal_evaluate_parameter_sets_batch(x)
        self.assert_close_arrays(res, np.zeros(25))
        self.assert_close_arrays(study._get_current_gradient_value(x), np.linspace(0,0.1, 25), 
                                 atol=1e-5)

    def test_matcal_evaluate_parameter_sets_batch_and_get_cur_hess_val(self):
        """no hessian for least squares"""

    def test_get_bounds(self):
        study = self._study_class(self.parameter_collection)
        bounds = study._get_bounds()
        self.assertEqual(([0.0], [10.0]), bounds)
        study = self._study_class(self.parameter_collection, method='lm')
        bounds = study._get_bounds()
        self.assertEqual((-np.inf, np.inf), bounds)


class TestAlgorithmOptions(MatcalUnitTest):

    def setUp(self):
        super().setUp(__file__)

    def test_algorithm_options(self):
        alg_opts = _AlgorithmOptions(False, False,False,False)
        self.assertEqual(alg_opts.gradient, False)
        self.assertEqual(alg_opts.hessian, False)
        self.assertEqual(alg_opts.bounds, False)
        self.assertEqual(alg_opts.constraints, False)

