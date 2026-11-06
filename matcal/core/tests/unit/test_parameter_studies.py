from collections import OrderedDict
from copy import deepcopy
from unittest.mock import patch

import numpy as np

from matcal.core.data import (convert_dictionary_to_data, DataCollection, 
                              ReturnPassedDataConditioner)
from matcal.core.models import PythonModel
from matcal.core.objective import (CurveBasedInterpolatedObjective, Objective, 
                                   SumSquaresMetricFunction, 
                                   SimulationResultsSynchronizer,
                                   DirectCurveBasedInterpolatedObjective)
                                   
from matcal.core.parameters import Parameter, ParameterCollection
from matcal.core.parameter_studies import (FiniteDifference, ClassicLaplaceStudy, 
                                           LaplaceStudy,
                                           ParameterStudy,
                                           HaltonStudy, 
                                           sample_multivariate_normal,
                                           _estimate_parameter_covariance, 
                                           _get_residual_covariance, 
                                           _combine_array_list_into_zero_padded_single_array, 
                                           _package_parameter_specific_results,
                                           _fit_posterior,
                                           _get_log_determinant_of_positive_definite_matrix,
                                           _log_posterior_predictive,
                                           _project_symmetric_matrix_to_psd,
                                           _solve_for_parameter_covariance,
                                           _solve_for_parameter_covariance_svd_subspace,
                                           _to_theta)

from matcal.core.state import State
from matcal.core.tests.MatcalUnitTest import MatcalUnitTest
from matcal.core.tests.unit.test_study_base import StudyBaseUnitTests, model_func


def linear_model(a, *args, **kwargs):
    x = 1
    y = a*x
    return {"x":x, "y":y}


def linear_model_with_length(a, *args, **kwargs):
    x = np.linspace(0,1,5)
    y = a*x 
    return {"x":x, "y":y}


def oneD_model(**param_dict):
    x = param_dict['theta']
    y = x ** 2
    return {"x":x, "f":y}


class FiniteDifferenceTest(MatcalUnitTest):

  def f(self,x,y,a0,a1,a2,a11,a12,a22):
    return a11*x*x+a22*y*y+a12*x*y+a1*x+a2*y+a0
  
  def f_cubic(self,x,y,a0,a1,a2,a11,a12,a22, a33, a34, a43, a44):
    return a43*y*y*x+a34*x*x*y+a44*y*y*y+a33*x*x*x+a11*x*x+a22*y*y+a12*x*y+a1*x+a2*y+a0

  def setUp(self):
    super().setUp(__file__)
  
  def test_consistency(self):
    a11 = 1.0
    a22 = 10.0
    a12 = 2.0
    a1 = 16.0
    a2 =  8.0
    a0 = 1.0
    parameters = [ a0,a1,a2,a11,a12,a22 ]
    optimum = [ -(a12*a2 - 2*a1*a22)/(a12*a12 - 4*a11*a22), 
               -((a1*a12 - 2*a11*a2)/(a12*a12 - 4*a11*a22)) ]
    finite_difference_operator = FiniteDifference(optimum,relative_step_size=1.e-3)
    points = finite_difference_operator.compute_hessian_evaluation_points()
    function_values = []
    for point in points:
      function_values.append(self.f(point[0],point[1], *parameters))
    finite_difference_operator.set_function_values(function_values)
    G = finite_difference_operator.gradient()
    Gref = np.zeros(G.shape)
    H = finite_difference_operator.hessian()
    Href = np.array([[2*a11,a12],[a12,2*a22]])
    self.assert_close_arrays(G,Gref)
    self.assert_close_arrays(H,Href)

  def test_results_ints_in(self):
    a11 = 3.2
    a22 = 1.5
    a12 = 4.35
    a1 = 3.1
    a2 =  2.4
    a0 = 1.0
    parameters = [ a0,a1,a2,a11,a12,a22 ]
    x=1000
    y=1
    finite_difference_operator = FiniteDifference([x,y] ,relative_step_size=1.e-3)
    points = finite_difference_operator.compute_hessian_evaluation_points()
    function_values = []
    for point in points:
      function_values.append(self.f(point[0],point[1], *parameters))
    finite_difference_operator.set_function_values(function_values)
    G = finite_difference_operator.gradient()
    Gref = np.array([2*a11*x+a12*y+a1, 2*a22*y+a12*x+a2])
    H = finite_difference_operator.hessian()
    Href = np.array([[2*a11,a12],[a12,2*a22]])

    self.assert_close_arrays(G,Gref)
    self.assert_close_arrays(H,Href, atol=1e-3)

  def test_results_cubic(self):
    a33=1.6
    a44=2.67
    a34=np.pi
    a43=np.e
    a11 = 2.3
    a22 = 6
    a12 = 3.5
    a1 = 0.26
    a2 =  3
    a0 = 10.0
    parameters = [ a0,a1,a2,a11,a12,a22, a33, a34, a43, a44]
    x=1
    y=600
    finite_difference_operator = FiniteDifference([x,y] ,relative_step_size=1.e-3)
    points = finite_difference_operator.compute_hessian_evaluation_points()
    function_values = []
    for point in points:
      function_values.append(self.f_cubic(point[0],point[1], *parameters))
    finite_difference_operator.set_function_values(function_values)
    G = finite_difference_operator.gradient()
    Gref = np.array([a43*y*y+2*a34*y*x+3*a33*x*x+2*a11*x+a12*y+a1, 
                     3*a44*y*y+2*a43*y*x+a34*x*x+2*a22*y+a12*x+a2])
    H = finite_difference_operator.hessian()
    Href = np.array([[2*a11+6*a33*x+2*a34*y,a12+2*a34*x+2*a43*y],
                     [2*a43*y+2*a34*x+a12,2*a22+2*a43*x+6*a44*y]])

    self.assert_close_arrays(G,Gref)
    self.assert_close_arrays(H,Href, atol=1e-3)

  def test_results_cubic_small(self):
    a33=1.6
    a44=2.67
    a34=np.pi
    a43=np.e
    a11 = 2.3
    a22 = 6
    a12 = 3.5
    a1 = 0.26
    a2 =  3
    a0 = 10.0
    parameters = [ a0,a1,a2,a11,a12,a22, a33, a34, a43, a44]
    x=0.0035
    y=0.001
    finite_difference_operator = FiniteDifference([x,y] ,relative_step_size=1.e-6, 
                                                  epsilon=np.finfo(float).eps**(1.0/3.0))
    points = finite_difference_operator.compute_hessian_evaluation_points()
    function_values = []
    for point in points:
      function_values.append(self.f_cubic(point[0],point[1], *parameters))
    finite_difference_operator.set_function_values(function_values)
    G = finite_difference_operator.gradient()
    Gref = np.array([a43*y*y+2*a34*y*x+3*a33*x*x+2*a11*x+a12*y+a1, 
                     3*a44*y*y+2*a43*y*x+a34*x*x+2*a22*y+a12*x+a2])
    H = finite_difference_operator.hessian()
    Href = np.array([[2*a11+6*a33*x+2*a34*y,a12+2*a34*x+2*a43*y],
                     [2*a43*y+2*a34*x+a12,2*a22+2*a43*x+6*a44*y]])

    self.assert_close_arrays(G,Gref)
    self.assert_close_arrays(H,Href, atol=1e-3)

  def test_results_cubic_small_grad_only(self):
    a33=1.6
    a44=2.67
    a34=np.pi
    a43=np.e
    a11 = 2.3
    a22 = 6
    a12 = 3.5
    a1 = 0.26
    a2 =  3
    a0 = 10.0
    parameters = [ a0,a1,a2,a11,a12,a22, a33, a34, a43, a44]
    x=0.0035
    y=0.001
    finite_difference_operator = FiniteDifference([x,y] ,relative_step_size=1.e-6, 
                                                  epsilon=np.finfo(float).eps**(1.0/3.0))
    points = finite_difference_operator.compute_gradient_evaluation_points()
    function_values = []
    for point in points:
      function_values.append(self.f_cubic(point[0],point[1], *parameters))
    finite_difference_operator.set_function_values(function_values)
    G = finite_difference_operator.gradient()
  
    Gref = np.array([a43*y*y+2*a34*y*x+3*a33*x*x+2*a11*x+a12*y+a1,
                      3*a44*y*y+2*a43*y*x+a34*x*x+2*a22*y+a12*x+a2])
    self.assert_close_arrays(G,Gref)

  def test_results_cubic_small_grad_only_forward_diff(self):
    a33=1.6
    a44=2.67
    a34=np.pi
    a43=np.e
    a11 = 2.3
    a22 = 6
    a12 = 3.5
    a1 = 0.26
    a2 =  3
    a0 = 10.0
    parameters = [ a0,a1,a2,a11,a12,a22, a33, a34, a43, a44]
    x=0.0035
    y=0.001
    finite_difference_operator = FiniteDifference([x,y] ,relative_step_size=1.e-6, 
                                                  epsilon=np.finfo(float).eps**(1.0/3.0))
    points = finite_difference_operator.compute_gradient_evaluation_points(three_point_finite_diff=False)
    function_values = []
    for point in points:
      function_values.append(self.f_cubic(point[0],point[1], *parameters))
    finite_difference_operator.set_function_values(function_values)
    G = finite_difference_operator.gradient()
  
    Gref = np.array([a43*y*y+2*a34*y*x+3*a33*x*x+2*a11*x+a12*y+a1, 
                     3*a44*y*y+2*a43*y*x+a34*x*x+2*a22*y+a12*x+a2])
    self.assert_close_arrays(G,Gref, atol=1e-4, rtol=1e-3)


class TestSampleMultivariateNorm(MatcalUnitTest):

    def setUp(self):
        super().setUp(__file__)
    def test_sample_covariance_bad(self):
        with self.assertRaises(TypeError):
            sample_multivariate_normal("bad input")
        with self.assertRaises(TypeError):
            sample_multivariate_normal(1.2, [1, 2], np.array([[1, 0], [0,1]]))
        with self.assertRaises(TypeError):
            sample_multivariate_normal(1, 1,  np.array([[1,0], [0,1]]))
        with self.assertRaises(ValueError):
            sample_multivariate_normal(1, [1, 1],  np.array([[], []]))
        with self.assertRaises(ValueError):
            sample_multivariate_normal(1, [1],  np.array([[1,0], [0,1]]))
        with self.assertRaises(TypeError):
            sample_multivariate_normal(1, [1, 0],  np.array([[1,0], [0,1]]), 
                                       seed=12.0)
        with self.assertRaises(ValueError):
            sample_multivariate_normal(1, [1, 0],  np.array([[1,0], [0,1]]), 
                                       seed=12, param_names=['a', 'b', 'c']) 

    def test_sample_covariance_good(self):
        mean = np.array([1,2])
        cov = np.array([[1,0], [0,1]])
        param_names = ['a', 'b']
        nsamples = 100
        samples = sample_multivariate_normal(100, mean, cov, 
                                             seed=12345, param_names=param_names)
        self.assertTrue(isinstance(samples, dict))
        self.assertTrue( 'a' in list(samples.keys()))
        self.assertTrue( 'b' in list(samples.keys()))
        
        self.assertEqual(nsamples, len(samples['a']))
        self.assertEqual(nsamples, len(samples['b']))

        samples2 = sample_multivariate_normal(100, mean, cov, 
                                             seed=12345)
        self.assertTrue(isinstance(samples, dict))
        self.assertTrue( 'parameter_0' in list(samples2.keys()))
        self.assertTrue( 'parameter_1' in list(samples2.keys()))
        self.assert_close_arrays(samples['a'], samples2['parameter_0'])
        samples2 = sample_multivariate_normal(100, mean, cov)
        self.assertFalse(np.allclose(samples['a'], samples2['parameter_0']))
        
                                               
class TestParameterStudy(StudyBaseUnitTests.CommonTests):

    _study_class = ParameterStudy

    def setUp(self):
        super().setUp(__file__)

    def _set_study_specific_options(self, study):
        study.add_parameter_evaluation(a=5)

    def test_set_restart_file_not_found(self):
        pass
    
    def test_write_restart_filename(self):
        pass

    def test_restart_custom_filename(self):
        pass

    def test_restart(self):
        pass

    def test_set_verbosity(self):
        pass

    def test_run_without_defined_parameter_evaluations(self):
        study = self._study_class(self.parameter_collection)
        study.add_evaluation_set(self.mock_model, self.objective, self.gold_results)
        with self.assertRaises(RuntimeError):
            study.launch()

    def test_add_parameter_evaluation(self):
            study = self._study_class(self.parameter_collection)
            study.add_parameter_evaluation(a=5)
            self.assertEqual(study.parameter_sets_to_evaluate[0], {"a":5})
        
    def test_add_multiple_parameter_evaluation(self):
        study = self._study_class(self.parameter_collection)
        study.add_parameter_evaluation(a=5)
        study.add_parameter_evaluation(a=6)
        pc1 = deepcopy(self.parameter_collection)
        pc1.update_parameters(a=5)
        pc1.set_name("1")
        pc2 = deepcopy(self.parameter_collection)
        pc2.update_parameters(a=6)
        pc2.set_name("2")
        gold_pcs = [pc1.get_current_value_dict(), pc2.get_current_value_dict()]
        for param_c_study, param_c_gold in zip(study.parameter_sets_to_evaluate, gold_pcs):
            self.assertEqual(param_c_study, param_c_gold),

    def test_add_parameter_evaluation_out_of_bounds(self):
        study = self._study_class(self.parameter_collection)

        with self.assertRaises(ValueError):
            study.add_parameter_evaluation(a=20)

    def test_incorrect_parameter_eval_requested(self):
        y = Parameter("Y", 50, 500)
        h = Parameter("H", 150, 1500, 200)
        e = Parameter("E", 100, 1000)
        nu = Parameter("nu", 0, .5, .33)

        PC = ParameterCollection("parameter_collections", y, h, e, nu)

        study = self._study_class(PC)
        with self.assertRaises(ValueError):
            study.add_parameter_evaluation(y=200, H=1000, nu=0.2)
        with self.assertRaises(ValueError):
            study.add_parameter_evaluation(Y=200, H=1000, E=100000, nu=0.2)

    def _get_basic_study(self, results_file=None, obj=None):
        y = Parameter("Y", 50, 500)
        h = Parameter("H", 150, 1500, 200)
        e = Parameter("E", 100, 1000)
        nu = Parameter("nu", 0, .5, .33)

        PC = ParameterCollection("parameter_collections", y, h, e, nu)

        study = self._study_class(PC)
        study.add_parameter_evaluation(Y=200, H=1000, E=500, nu=0.2)
        study.add_parameter_evaluation(Y=201, H=200, E=1000, nu=0.5)

        if results_file is not None:
            self.mock_model.set_results_filename(results_file)          
        if obj is None:
            obj = self.objective
        study.add_evaluation_set(self.mock_model, obj, self.gold_results)
        param_batch_evaluator = study._initialize_study_and_batch_evaluator()    
        return study, PC

    def _get_mult_model_mult_state_study(self, results_file=None):
        y = Parameter("Y", 50, 500)
        h = Parameter("H", 150, 1500, 200)
        e = Parameter("E", 100, 1000)
        nu = Parameter("nu", 0, .5, .33)

        PC = ParameterCollection("parameter_collections", y, h, e, nu)

        study = self._study_class(PC)
        study.add_parameter_evaluation(Y=200, H=1000, E=500, nu=0.2)
        study.add_parameter_evaluation(Y=201, H=200, E=1000, nu=0.5)

        study.add_evaluation_set(self.mock_model, self.objective, self.data_collection2)
        study.add_evaluation_set(self.mock_model2, self.objective, self.data_collection)

        param_batch_evaluator = study._initialize_study_and_batch_evaluator()    
        return study, PC

    def test_make_residuals_study(self):
        study, data = self._get_actual_param_study()
        self.assertTrue(study._return_residuals)
        study.make_total_objective_study()
        self.assertFalse(study._return_residuals)
        study.make_residuals_study()
        self.assertTrue(study._return_residuals)
        def normalize(data):
            return data/12.0
        results = study.launch()
        error = normalize(data["y"]) - normalize(model_func(5.0, 0.5, 2.5)["y"])
        residual_total_obj = np.linalg.norm(np.hstack((error/np.sqrt(len(error)), 
                                                       error/np.sqrt(len(error)))))
        eval_set = results.evaluation_sets[0]
        self.assertEqual(results.total_objective_history[-1], 
                         residual_total_obj**2)

    def test_make_total_objective_study(self):
        study, data = self._get_actual_param_study()
        self.assertTrue(study._return_residuals)
        study.make_total_objective_study()
        self.assertFalse(study._return_residuals)
        def normalize(data):
            return data/12
        results = study.launch()
        error = normalize(data["y"]) - normalize(model_func(5.0, 0.5, 2.5)["y"])
        obj_total_obj = 2*np.linalg.norm(error/np.sqrt(len(error)))**2
        eval_set = results.evaluation_sets[0]

        self.assertEqual(results.total_objective_history[-1], obj_total_obj)

    def _get_actual_param_study(self):
        coeff = Parameter("coeff", 0, 10)
        power = Parameter("power", 0, 1)
        offset = Parameter("offset", 0, 5)
        pc = ParameterCollection("test", coeff, power, offset)
        zero_obj_data = model_func(1.0, 1.0, 2.0)
        data = convert_dictionary_to_data(zero_obj_data)
        study = self._study_class(pc)
        power_model = PythonModel(model_func)
        obj = CurveBasedInterpolatedObjective("x", "y")
        obj.set_name("test_obj")
        obj2 = Objective("y")
        obj2.set_name("test_obj2")

        study.add_parameter_evaluation(**pc.get_current_value_dict())
        study.add_evaluation_set(power_model, 
                                 obj, data)
        study.add_evaluation_set(power_model, 
                                 obj2, data)

        return study, data

    def test_set_run_async_to_false(self):
        study, data = self._get_actual_param_study()
        self.assertTrue(study._run_async)
        study.run_in_serial()
        self.assertFalse(study._run_async)


class TestHaltonStudy(StudyBaseUnitTests.CommonTests):

    _study_class = HaltonStudy

    @staticmethod
    def setup_1d_parameter_collection():
        theta = Parameter("theta", -2, 2, distribution="uniform_uncertain")
        return ParameterCollection("one_parameter", theta)
        
    @staticmethod
    def run_study(study, nsamples, model_name, par_names, skip=None):
        study.set_number_of_samples(nsamples, skip)
        results = study.launch()
        params = np.array([results.parameter_history[par] for par in par_names]).T.squeeze()
        state0 = results.simulation_history[model_name].states['matcal_default_state']
        sim_history = results.simulation_history[model_name][state0]
        return params, sim_history, state0

    @staticmethod
    def calculate_interpolated_pred_error(params, data, test_points, test_data):
        from scipy.interpolate import interp1d
        sorting_indices = np.argsort(params)
        sorted_params = np.array(params)[sorting_indices]
        sorted_data = np.array(data)[sorting_indices]
         
        f = interp1d(sorted_params, sorted_data, kind='linear', fill_value='extrapolate')
        y_pred = f(test_points)
        return np.linalg.norm(y_pred - test_data), y_pred
            
    def setup_study(self, scramble=False): 
        self.model_name = 'oneD'
        self.par_names = ['theta']
        
        # set up model, objective, parameter collection
        self.model = PythonModel(oneD_model)
        self.model.set_name(self.model_name)
        parameter_collection = TestHaltonStudy.setup_1d_parameter_collection()
        test_points = np.linspace(-2, 2, 5) 
        objective = SimulationResultsSynchronizer("x", test_points, "f")
        self.study = self._study_class(parameter_collection)
        self.study.set_Halton_scramble(scramble)
        self.study.set_seed(42)
        self.study.add_evaluation_set(self.model, objective)
    
    def setUp(self):
        super().setUp(__file__)


    def test_check_variable_type(self):
        pass
    
    def test_set_number_of_samples(self):
        pass
    
    def test_generate_samples(self):
        pass
        
    def test_populate_parameter_evaluations(self):
        pass        

    def test_skip_ahead(self):
        pass
    
    def test_scale_samples_to_bounds(self):
        pass
    
    def test_1d_launch(self):
        # run initial study
        nsamples = 10
        self.setup_study()
        params, sim_history, state0 = TestHaltonStudy.run_study(self.study, nsamples, self.model_name, self.par_names)
        self.assertEqual(len(params), nsamples)
        
        data = [sim_history[i]['f'][0] for i in range(nsamples)]

        # evaluate model at test points
        test_points = np.linspace(-2, 2)
        test_data = []
        import time
        for val in test_points: 
            th = Parameter("theta", -2, 2, distribution="uniform_uncertain", current_value=val)
            pc = ParameterCollection("predictions", th)
            res = self.model.run(state0, pc)
            test_data.append(res.results_data['f'][0])
            time.sleep(0.1)
        test_data = np.array(test_data)
         
        # interpolate and calculate prediction error of test points
        pred_error, _ = TestHaltonStudy.calculate_interpolated_pred_error(\
            params, data, test_points, test_data)

        # continue study with additional Halton samples
        nnew_samples = 12
        self.setup_study()
        self.study.restart()
        new_params, sim_history, state0 = TestHaltonStudy.run_study(self.study, nsamples+nnew_samples, self.model_name, self.par_names)
        self.assertEqual(len(new_params), nsamples + nnew_samples)

        new_data = [sim_history[i]['f'][0] for i in range(nsamples+nnew_samples)]

        # interpolate and calculate prediction error of test points
        new_pred_error, _ = TestHaltonStudy.calculate_interpolated_pred_error(\
            new_params, new_data, test_points, test_data)
        
        # prediction error should be less with more Halton samples
        self.assertGreater(pred_error, new_pred_error)

    def test_error_handling(self):    
        # check variable type error handling
        nsamples = 10
        self.setup_study()
        with self.assertRaises(TypeError):
            TestHaltonStudy.run_study(self.study, nsamples, self.model_name, self.par_names, skip=True)
        
        with self.assertRaises(RuntimeError):
            self.study.add_parameter_evaluation(a=5)
        
    def test_skipping(self):
        # skip not None
        nsamples = 10
        self.setup_study()
        params, _, _= TestHaltonStudy.run_study(self.study, nsamples, self.model_name, self.par_names, skip=10)
        self.assertEqual(len(params), nsamples)
       
        
def model_linear(a):
    disp = np.linspace(0,1, 10)
    load = a*disp
    return {"displacement":disp, "load":load}

class TestLaplaceStudy(StudyBaseUnitTests.CommonTests):

    _study_class = LaplaceStudy

    def setUp(self):
        super().setUp(__file__)

    def _set_study_specific_options(self, study):
        study.set_parameter_center(a=5)

    def test_set_restart_file_not_found(self):
        pass
    
    def test_write_restart_filename(self):
        pass

    def test_restart_custom_filename(self):
        pass

    def test_restart(self):
        pass

    def test_set_verbosity(self):
        pass

    def test_launching_a_study_twice_raises_error(self):
        pass

    def test_add_parameter_evaluation(self):
        study = self._study_class(self.parameter_collection)
        with self.assertRaises(study.StudyInputError):
            study.add_parameter_evaluation(a=5)
    
    def test_get_parameter_center_index(self):
        study = self._study_class(self.parameter_collection)
        self.assertEqual(study._get_center_eval_index(), 0)

    def test_run_without_center(self):
        study = self._study_class(self.parameter_collection)
        study.add_evaluation_set(self.mock_model, self.objective, self.gold_results)
        with self.assertRaises(RuntimeError):
            study.launch()
            
    def test_set_step_size(self):
        study = self._study_class(self.parameter_collection)
        study.set_parameter_center(a=2)

        self.assertEqual(study._step_size, 1e-3)
        self.assertEqual(study._finite_difference._relative_step_size, 1e-3)

        study.set_step_size(1e-4)
        self.assertEqual(study._step_size, 1e-4)
        study.set_parameter_center(a=2)
        self.assertEqual(study._finite_difference._relative_step_size, 1e-4)
        with self.assertRaises(TypeError):
            study.set_step_size("a")

        with self.assertRaises(ValueError):
            study.set_step_size(0)

        with self.assertRaises(ValueError):
            study.set_step_size(1)

    def test_bad_objectives(self):
        bad_data_dict = {"x":[0, 1]}
        bad_data = convert_dictionary_to_data(bad_data_dict)
        good_data_dict = {"x":[0]}
        good_data = convert_dictionary_to_data(good_data_dict)

        bad_data_dict_2 = {"x":[2 ,3]}
        bad_data_2 = convert_dictionary_to_data(bad_data_dict_2)
        bad_data_2.set_state(State("bad"))
        
        study = self._study_class(self.parameter_collection)

        py_mod = PythonModel(model_linear)
        py_mod.set_name("py_mod")
        obj1 = Objective("x")

        with self.assertRaises(ValueError):
            study.add_evaluation_set(py_mod, obj1, bad_data)
        
        study._evaluation_sets = OrderedDict()
        bad_repeat_dc = DataCollection("test", good_data, bad_data)
        with self.assertRaises(ValueError):
            study.add_evaluation_set(py_mod, obj1, bad_repeat_dc)

        study._evaluation_sets = OrderedDict()
        bad_state_dc = DataCollection("test", good_data, bad_data_2)
        with self.assertRaises(ValueError):
            study.add_evaluation_set(py_mod, obj1, bad_state_dc)

    def test_get_overall_results(self):
        study = self._study_class(self.parameter_collection)
        overall_results = study._get_overall_results({"cov":1})
        self.assertTrue("parameter_order" in overall_results)
        self.assertTrue("cov" in overall_results)
        self.assertEqual(overall_results["cov"], 1)
        self.assertEqual(overall_results["parameter_order"], ['a'])       
    
    def test_combine_array_list_into_zero_padded_array(self):
        repeats = (np.ones((5,2)), 2*np.ones((10,5)), 3*np.ones((4,3)))
        
        combined_array = _combine_array_list_into_zero_padded_single_array(repeats)
        gold_array = np.zeros((19, 5))
        gold_array[0:5, 0:2] = repeats[0]
        gold_array[5:15, :] = repeats[1]
        gold_array[15:, 0:3] = repeats[2]
        
        self.assert_close_arrays(combined_array, gold_array)

    def test_get_parameter_specific_results_no_length(self):
        study = self._study_class(self.parameter_collection)
        study.set_parameter_center(a=1)
        model = PythonModel(linear_model)
        data1 = convert_dictionary_to_data({"x":1,"y":1})
        data2 = convert_dictionary_to_data({"x":1,"y":1.01})
        data3 = convert_dictionary_to_data({"x":1,"y":1.02})
        dc = DataCollection("test", data1, data2, data3)

        study.add_evaluation_set(model, Objective("y"), dc, 
            data_conditioner_class=ReturnPassedDataConditioner)
        res = study.launch()

        study_param_results = study._get_parameter_specific_results("grad_key")
        self.assertEqual(study_param_results["mean:a"], 1)
        #for raw residuals just one, for scaled residuals divide by sqrt of 3 for 
        # normalization by number of data sets
        self.assert_close_arrays(study_param_results["grad_key:a"], np.ones((1,3)))
        
        self.assert_close_arrays(res.residuals_gradient.a, np.ones((1,3)))
        self.assertEqual(res.parameter_order, ['a'])
        self.assert_close_dicts_or_data(res.residuals_gradient.to_dict(), 
                                        {'a':np.ones((1,3))})

    def test_get_parameter_specific_results_with_length(self):
        study = self._study_class(self.parameter_collection)
        study.set_parameter_center(a=1)
        model = PythonModel(linear_model_with_length)
        data1 = convert_dictionary_to_data(linear_model_with_length(1))
        data2 = convert_dictionary_to_data(linear_model_with_length(1.01))
        data3 = convert_dictionary_to_data(linear_model_with_length(1.02))
        dc = DataCollection("test", data1, data2, data3)

        study.add_evaluation_set(model, DirectCurveBasedInterpolatedObjective("x","y"), 
            dc, data_conditioner_class=ReturnPassedDataConditioner)
        res = study.launch()

        study_param_results = study._get_parameter_specific_results("grad_key")
        self.assertEqual(study_param_results["mean:a"], 1)
        lin_x = np.linspace(0,1,5)
        gold_grad = np.array([lin_x, lin_x, lin_x]).T
        self.assert_close_arrays(study_param_results["grad_key:a"], gold_grad, show_on_fail=True)

    def test_set_calibrate_covariance(self):
        study = self._study_class(self.parameter_collection)
        study.set_parameter_center(a=1)
        self.assertTrue(study._calibrate_covariance)
        study.set_calibrate_covariance(False)
        self.assertFalse(study._calibrate_covariance)
        model = PythonModel(linear_model_with_length)
        data1 = convert_dictionary_to_data(linear_model_with_length(1))
        data2 = convert_dictionary_to_data(linear_model_with_length(1.01))
        data3 = convert_dictionary_to_data(linear_model_with_length(1.02))
        dc = DataCollection("test", data1, data2, data3)

        study.add_evaluation_set(model, DirectCurveBasedInterpolatedObjective("x","y"), 
            dc, data_conditioner_class=ReturnPassedDataConditioner)
        res = study.launch()
        with self.assertRaises(AttributeError):
            res.fitted_parameter_covariance
        res.estimated_parameter_covariance
        study_param_results = study._get_parameter_specific_results("grad_key")
        self.assertEqual(study_param_results["mean:a"], 1)
        lin_x = np.linspace(0,1,5)
        gold_grad = np.array([lin_x, lin_x, lin_x]).T
        self.assert_close_arrays(study_param_results["grad_key:a"], gold_grad, show_on_fail=True)

    def test_package_parameter_specific_results(self):
        param_collect = {'a':None, 'b':None}
        sens_info = {"param_dep":[0,1], "param_independent":0}
        packaged_data = _package_parameter_specific_results(param_collect, sens_info)
        self.assertTrue("param_dep:a" in packaged_data)
        self.assertTrue("param_dep:b" in packaged_data)
        self.assertFalse("param_independent:a" in packaged_data)
        self.assertFalse("param_independent" in packaged_data)
        self.assertEqual(packaged_data["param_dep:a"], 0)
        self.assertEqual(packaged_data["param_dep:b"], 1)


class TestEstimateParameterCovariance(MatcalUnitTest):
    def setUp(self):
        super().setUp(__file__)

    def test_no_repeats(self):
        resids = np.zeros(10)
        with self.assertRaises(RuntimeError):
            _estimate_parameter_covariance(resids, resids, None)
        
    def test_get_residual_covariance_correlated(self):
        std = 0.5
        var = std*std
        resids = np.random.normal(scale = std, size=(10,1000))
        resids_cov = _get_residual_covariance(resids)
        self.assert_close_arrays(resids_cov, np.cov(resids.T), atol=3e-3, 
                                 show_on_fail=True)

    def test_estimate_parameter_covariance_linear_correlated(self):
        mean = np.array([10, 20])
        cov =  np.array([[0.05, 0.02], [0.02, 0.03]])
        inputs = sample_multivariate_normal(4000000, mean, 
                                           cov, seed=4123,
                                            param_names=["a", "b"])
        def model(inputs):
            x = np.linspace(0.1,2,10)
            return inputs["a"]*x[:, np.newaxis]+inputs["b"]
        
        def resid_sensitivity(inputs):
            x = np.linspace(0.1,2,10)
            return np.array([-x, -np.ones(len(x))])
        
        mean_dict = {"a":mean[0], "b":mean[1]}
        resids = model(mean_dict) - model(inputs)
        resids = resids.T
        sens = resid_sensitivity(mean_dict).T
        cov_est = _estimate_parameter_covariance(resids, sens, 0)
        self.assert_close_arrays(cov, cov_est, show_on_fail=True, 
                                 rtol=1e-3)
        
    def test_estimate_parameter_covariance_under_determined(self):
        mean = np.array([10, 20])
        cov =  np.array([[1e-13, 1e-13], [1e-13, 1e-13]])
        inputs = sample_multivariate_normal(2, mean, 
                                           cov, seed=4123,
                                            param_names=["a", "b"])
        def model(inputs):
            x = np.linspace(0.1,2,10)
            return inputs["a"]*x[:, np.newaxis]+inputs["b"]
        
        def resid_sensitivity(inputs):
            x = np.linspace(0.1,2,10)
            return np.array([-x, -np.ones(len(x))])
        
        mean_dict = {"a":mean[0], "b":mean[1]}
        resids = model(mean_dict) - model(inputs)
        resids = resids.T
        sens = resid_sensitivity(mean_dict).T
        with self.assertRaises(ValueError):
            cov_est = _estimate_parameter_covariance(resids, sens, 0)
        

class TestFitPosteriors(MatcalUnitTest):
    def setUp(self):
        super().setUp(__file__)

    def test_estimate_parameter_covariance_linear_correlated(self):
        mean = np.array([10, 20])
        cov =  np.array([[0.05, 0.02], [0.02, 0.03]])
        n_repeats=100000
        inputs = sample_multivariate_normal(n_repeats, mean, 
                                           cov, seed=4123,
                                            param_names=["a", "b"])
        
        def model(inputs):
            x = np.linspace(0.1,2,10)
            return inputs["a"]*x[:, np.newaxis]*x[:, np.newaxis]*x[:, np.newaxis]+inputs["b"]
        
        def resid_sensitivity(inputs):
            x = np.linspace(0.1,2,10)
            return np.array([-x**3, -np.ones(len(x))])
        
        mean_dict = {"a":mean[0], "b":mean[1]}

        std = 1e-2
        np.random.seed(10)
        noise = np.random.normal(scale = std, size=(10,n_repeats))

        resids = model(mean_dict) - model(inputs)
        resids += noise
       
        resids = resids.T
        noise_guess = std**2
        sens = resid_sensitivity(mean_dict).T
        cov_est = _estimate_parameter_covariance(resids, sens, noise_guess)
        start = np.copy(cov_est)
        fitted_posterior = _fit_posterior(resids, sens, start, noise_guess, method=None)
        self.assert_close_arrays(cov, fitted_posterior, show_on_fail=True, 
                                 rtol=1e-2)


class TestClassicLaplaceStudy(StudyBaseUnitTests.CommonTests):

    _study_class = ClassicLaplaceStudy

    def setUp(self):
        super().setUp(__file__)

    def _set_study_specific_options(self, study):
        study.set_parameter_center(a=5)

    def test_set_restart_file_not_found(self):
        pass
    
    def test_write_restart_filename(self):
        pass

    def test_restart_custom_filename(self):
        pass

    def test_restart(self):
        pass

    def test_set_verbosity(self):
        pass

    def test_add_parameter_evaluation(self):
        study = self._study_class(self.parameter_collection)
        with self.assertRaises(study.StudyInputError):
            study.add_parameter_evaluation(a=5)
    
    def test_run_without_center(self):
        study = self._study_class(self.parameter_collection)
        study.add_evaluation_set(self.mock_model, self.objective, self.gold_results)
        with self.assertRaises(RuntimeError):
            study.launch()
            
    def test_set_step_size(self):
        study = self._study_class(self.parameter_collection)
        study.set_parameter_center(a=2)
        study._setup_finite_difference()
        self.assertEqual(study._step_size, 1e-3)
        self.assertEqual(study._finite_difference._relative_step_size, 1e-3)

        study.set_step_size(1e-4)
        self.assertEqual(study._step_size, 1e-4)
        study.set_parameter_center(a=2)
        self.assertEqual(study._finite_difference._relative_step_size, 1e-4)
        with self.assertRaises(TypeError):
            study.set_step_size("a")

        with self.assertRaises(ValueError):
            study.set_step_size(0)

        with self.assertRaises(ValueError):
            study.set_step_size(1)

    def test_results_no_length(self):
        study = self._study_class(self.parameter_collection)
        study.set_parameter_center(a=1)
        model = PythonModel(linear_model)
        data1 = convert_dictionary_to_data({"x":1,"y":1})
        dc = DataCollection("test", data1)

        study.add_evaluation_set(model, Objective("y"), dc, 
                                 data_conditioner_class=ReturnPassedDataConditioner)
        res = study.launch()

        self.assertAlmostEqual(res.objective_gradient.a, 0)
        self.assertEqual(res.parameter_order, ['a'])
        self.assert_close_dicts_or_data(res.objective_gradient.to_dict(), {'a':0})
        self.assertAlmostEqual(res.hessian[0,0], 2)
 
    def test_results_with_length(self):
        study = self._study_class(self.parameter_collection)
        study.set_parameter_center(a=1)
        study.set_step_size(1e-6)
        model = PythonModel(linear_model_with_length)
        data1 = convert_dictionary_to_data(linear_model_with_length(1))
        dc = DataCollection("test", data1)
        obj = CurveBasedInterpolatedObjective("x","y")
        obj.set_metric_function(SumSquaresMetricFunction())
        study.add_evaluation_set(model, obj, data1, 
                                 data_conditioner_class=ReturnPassedDataConditioner)
        res = study.launch()

        self.assertAlmostEqual(res.objective_gradient.a, 0)
        self.assertEqual(res.parameter_order, ['a'])
        self.assert_close_dicts_or_data(res.objective_gradient.to_dict(), {'a':0})
        #obj = (a*x - data)**2/norm_fact**2
        #dobj/da = 2*(a*x-data)*x
        #ddobj/dda = sum(2*x**2)
        goal = np.sum(2*np.linspace(0,1,5)**2)/5
        self.assertAlmostEqual(res.hessian[0,0], goal)


def _vector_residuals(point):
    """Residuals with known Hessians; shape n_residuals x n_repeats."""
    a, b = point
    base = np.array([a * a * b, np.sin(a) * b, a + b])
    return np.stack([base, 2 * base], axis=1)


def _vector_residual_hessians(point):
    """Analytic Hessians of the first repeat of _vector_residuals."""
    a, b = point
    hessians = np.zeros((3, 2, 2))
    hessians[0] = [[2 * b, 2 * a], [2 * a, 0]]
    hessians[1] = [[-np.sin(a) * b, np.cos(a)], [np.cos(a), 0]]
    return hessians


def _make_two_parameter_laplace_study():
    return LaplaceStudy(Parameter("a", 0, 10), Parameter("b", 0, 10))


class TestLaplaceStudyHessianOptions(MatcalUnitTest):
    def setUp(self):
        super().setUp(__file__)
        self.center = np.array([1.3, 0.7])
        self.study = _make_two_parameter_laplace_study()

    def _set_center_and_function_values(self, residual_function):
        self.study.set_parameter_center(a=self.center[0], b=self.center[1])
        finite_difference = self.study._finite_difference
        points = finite_difference._finite_difference_evaluation_points
        finite_difference.set_function_values([residual_function(p) for p in points])

    def test_set_use_hessian_switches_evaluation_point_sets(self):
        self.study.set_parameter_center(a=1, b=2)
        self.assertEqual(len(self.study.parameter_sets_to_evaluate), 5)
        self.study.set_use_hessian(True)
        self.assertEqual(len(self.study.parameter_sets_to_evaluate), 9)
        self.study.set_use_hessian(False)
        self.assertEqual(len(self.study.parameter_sets_to_evaluate), 5)

    def test_set_use_hessian_before_center_and_bad_input(self):
        self.study.set_use_hessian(True)
        self.assertTrue(self.study._use_hessian)
        with self.assertRaises(TypeError):
            self.study.set_use_hessian("yes")
        with self.assertRaises(TypeError):
            self.study.set_use_hessian(use_hesssian=True)

    def test_residual_hessians_with_repeats_match_analytic(self):
        self.study.set_use_hessian(True)
        self._set_center_and_function_values(_vector_residuals)
        hessians = self.study._calculate_residual_hessians()
        self.assert_close_arrays(hessians, _vector_residual_hessians(self.center),
                                 atol=1e-5)

    def test_residual_hessians_without_repeats_match_analytic(self):
        self.study.set_use_hessian(True)
        self._set_center_and_function_values(lambda p: _vector_residuals(p)[:, 0])
        hessians = self.study._calculate_residual_hessians()
        self.assert_close_arrays(hessians, _vector_residual_hessians(self.center),
                                 atol=1e-5)

    def test_residual_hessian_for_scalar_residual(self):
        self.study.set_use_hessian(True)
        self._set_center_and_function_values(lambda p: _vector_residuals(p)[0, 0])
        hessians = self.study._calculate_residual_hessians()
        self.assertEqual(hessians.shape, (1, 2, 2))
        self.assert_close_arrays(hessians[0], _vector_residual_hessians(self.center)[0],
                                 atol=1e-5)

    def test_residual_hessians_require_hessian_points(self):
        self._set_center_and_function_values(_vector_residuals)
        with self.assertRaises(RuntimeError):
            self.study._calculate_residual_hessians()

    def test_residual_hessians_reject_unexpected_shape(self):
        self.study.set_use_hessian(True)
        self._set_center_and_function_values(lambda p: np.ones((3, 2, 2)))
        with self.assertRaises(RuntimeError):
            self.study._calculate_residual_hessians()

    def test_gauss_newton_objective_hessian(self):
        sensitivities = np.array([[1.0, 2.0], [0.5, -1.0], [3.0, 0.0]])
        center_residuals = np.ones((2, 3))
        hessian = self.study._calculate_objective_hessian(center_residuals,
                                                          sensitivities)
        self.assert_close_arrays(hessian, 2 * sensitivities.T @ sensitivities)
        with self.assertRaises(RuntimeError):
            self.study._calculate_gauss_newton_objective_hessian(np.ones(3))

    def test_full_objective_hessian_adds_mean_residual_term(self):
        self.study.set_use_hessian(True)
        self._set_center_and_function_values(_vector_residuals)
        sensitivities = np.array([[1.0, 2.0], [0.5, -1.0], [3.0, 0.0]])
        center_residuals = np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
        hessian = self.study._calculate_objective_hessian(center_residuals,
                                                          sensitivities)
        mean_residuals = np.array([2.0, 2.0, 2.0])
        second_order = 2 * np.einsum("k,kij->ij", mean_residuals,
                                     _vector_residual_hessians(self.center))
        goal = 2 * sensitivities.T @ sensitivities + second_order
        self.assert_close_arrays(hessian, goal, atol=1e-4)

    def test_center_residual_vector_shapes(self):
        get_vector = self.study._get_center_residual_vector_for_objective_hessian
        self.assert_close_arrays(get_vector(np.array([1.0, 2.0])), [1.0, 2.0])
        self.assert_close_arrays(get_vector(np.array([[1.0, 2.0], [3.0, 4.0]])),
                                 [2.0, 3.0])
        with self.assertRaises(RuntimeError):
            get_vector(np.ones((2, 2, 2)))

    def test_second_order_term_rejects_bad_shapes(self):
        second_order = self.study._calculate_second_order_residual_hessian_term
        with self.assertRaises(RuntimeError):
            second_order(np.ones(3), np.ones((3, 2)))
        with self.assertRaises(RuntimeError):
            second_order(np.ones((3, 1)), np.ones((3, 2, 2)))
        with self.assertRaises(RuntimeError):
            second_order(np.ones(2), np.ones((3, 2, 2)))


class TestLaplaceStudyCovarianceOptions(MatcalUnitTest):
    def setUp(self):
        super().setUp(__file__)
        self.study = _make_two_parameter_laplace_study()

    def test_default_initial_covariance_estimator_is_least_squares(self):
        self.assertEqual(self.study._initial_covariance_method, "least_squares")

    def test_set_initial_covariance_estimator(self):
        self.study.set_initial_covariance_estimator("LS", rcond=1e-8,
                                                    regularization=0.1,
                                                    project_to_psd=True,
                                                    min_eigenvalue=1e-6)
        self.assertEqual(self.study._initial_covariance_method, "least_squares")
        self.assertEqual(self.study._initial_covariance_rcond, 1e-8)
        self.assertEqual(self.study._initial_covariance_regularization, 0.1)
        self.assertTrue(self.study._initial_covariance_project_to_psd)
        self.assertEqual(self.study._initial_covariance_min_eigenvalue, 1e-6)
        self.study.set_initial_covariance_estimator("svd_pinv")
        self.assertEqual(self.study._initial_covariance_method, "svd_pinv")

    def test_set_initial_covariance_estimator_bad_input(self):
        with self.assertRaises(ValueError):
            self.study.set_initial_covariance_estimator("cholesky")
        with self.assertRaises(ValueError):
            self.study.set_initial_covariance_estimator(rcond=-1.0)
        with self.assertRaises(TypeError):
            self.study.set_initial_covariance_estimator(project_to_psd=1)

    def test_set_calibrate_covariance_keyword(self):
        self.study.set_calibrate_covariance(calibrate_covariance=False)
        self.assertFalse(self.study._calibrate_covariance)
        with self.assertRaises(TypeError):
            self.study.set_calibrate_covariance(calibrate_covaraince=False)

    def _make_study_with_repeats(self, number_of_repeats):
        study = LaplaceStudy(Parameter("a", 0, 10))
        study.set_parameter_center(a=1)
        data = [convert_dictionary_to_data(linear_model_with_length(1 + 0.01 * i))
                for i in range(number_of_repeats)]
        study.add_evaluation_set(PythonModel(linear_model_with_length),
                                 DirectCurveBasedInterpolatedObjective("x", "y"),
                                 DataCollection("test", *data),
                                 data_conditioner_class=ReturnPassedDataConditioner)
        return study

    def test_launch_fails_before_evaluations_with_too_few_repeats(self):
        study = self._make_study_with_repeats(1)
        self.assertEqual(study._get_maximum_number_of_repeats(), 1)
        with patch.object(LaplaceStudy, "_matcal_evaluate_parameter_sets_batch") as run:
            with self.assertRaises(ValueError):
                study.launch()
        run.assert_not_called()

    def test_repeat_check_skipped_for_svd_pinv(self):
        study = self._make_study_with_repeats(1)
        study.set_initial_covariance_estimator("svd_pinv")
        study._check_enough_repeats_before_evaluations()

    def test_repeat_check_passes_with_enough_repeats(self):
        study = self._make_study_with_repeats(3)
        self.assertEqual(study._get_maximum_number_of_repeats(), 3)
        study._check_enough_repeats_before_evaluations()

    def test_update_laplace_estimate_requires_launch(self):
        with self.assertRaises(RuntimeError):
            self.study.update_laplace_estimate(0.1)

    def test_update_laplace_estimate_after_launch(self):
        study = self._make_study_with_repeats(3)
        study.set_calibrate_covariance(False)
        study.launch()
        results = study.update_laplace_estimate(0.001)
        self.assertAlmostEqual(study._noise_variance, 1e-6)
        self.assertEqual(results.estimated_parameter_covariance.shape, (1, 1))


class TestParameterCovarianceEstimators(MatcalUnitTest):
    def setUp(self):
        super().setUp(__file__)
        rng = np.random.default_rng(1234)
        self.sensitivities = rng.normal(size=(6, 3))
        self.parameter_covariance = np.array([[2.0, 0.3, 0.1],
                                              [0.3, 1.0, 0.2],
                                              [0.1, 0.2, 0.5]])
        self.noise_variance = 0.01
        self.output_covariance = (self.sensitivities @ self.parameter_covariance
                                  @ self.sensitivities.T
                                  + self.noise_variance * np.eye(6))

    def test_all_methods_recover_exact_covariance(self):
        for method in ["least_squares", "ls", "svd_pinv", "svd_inverse"]:
            estimate = _solve_for_parameter_covariance(
                self.output_covariance, self.sensitivities, self.noise_variance,
                method=method)
            self.assert_close_arrays(estimate, self.parameter_covariance,
                                     atol=1e-10, show_on_fail=True)

    def test_unknown_method_raises(self):
        with self.assertRaises(ValueError):
            _solve_for_parameter_covariance(self.output_covariance,
                                            self.sensitivities, method="qr")

    def _collinear_problem(self):
        sensitivities = self.sensitivities.copy()
        sensitivities[:, 2] = sensitivities[:, 0] + sensitivities[:, 1]
        output_covariance = (sensitivities @ np.diag([1.0, 2.0, 3.0])
                             @ sensitivities.T + 1e-3 * np.eye(6))
        return output_covariance, sensitivities

    def test_least_squares_raises_for_unidentifiable_parameters(self):
        output_covariance, sensitivities = self._collinear_problem()
        with self.assertRaisesRegex(ValueError, "Unidentifiable directions"):
            _solve_for_parameter_covariance(output_covariance, sensitivities)

    def test_regularization_or_pinv_accept_unidentifiable_parameters(self):
        output_covariance, sensitivities = self._collinear_problem()
        regularized = _solve_for_parameter_covariance(
            output_covariance, sensitivities, regularization=1e-6)
        pseudo_inverse = _solve_for_parameter_covariance(
            output_covariance, sensitivities, method="svd_pinv")
        self.assertEqual(regularized.shape, (3, 3))
        self.assertEqual(pseudo_inverse.shape, (3, 3))

    def test_least_squares_raises_without_scatter(self):
        residuals = np.tile(np.linspace(0, 1, 6), (10, 1))
        with self.assertRaisesRegex(ValueError, "too little scatter"):
            _estimate_parameter_covariance(residuals, self.sensitivities, 0.0)

    def test_least_squares_raises_with_too_few_repeats(self):
        residuals = np.random.default_rng(5).normal(size=(3, 6))
        with self.assertRaisesRegex(ValueError, "at least 4 repeats"):
            _estimate_parameter_covariance(residuals, self.sensitivities, 0.0)

    def test_svd_inverse_raises_for_zero_singular_values(self):
        with self.assertRaises(ValueError):
            _solve_for_parameter_covariance(np.zeros((6, 6)), self.sensitivities,
                                            method="svd_inverse")

    def test_svd_subspace_rejects_unsupported_method(self):
        with self.assertRaises(ValueError):
            _solve_for_parameter_covariance_svd_subspace(
                self.output_covariance, self.sensitivities, method="least_squares")

    def test_svd_methods_raise_with_fewer_outputs_than_parameters(self):
        with self.assertRaises(ValueError):
            _solve_for_parameter_covariance(np.eye(2), np.ones((2, 3)),
                                            method="svd_pinv")

    def test_project_to_psd_option(self):
        estimate = _solve_for_parameter_covariance(-np.eye(3), np.eye(3),
                                                   method="svd_pinv",
                                                   project_to_psd=True,
                                                   min_eigenvalue=1e-6)
        self.assertGreaterEqual(np.linalg.eigvalsh(estimate).min(), 1e-6 * 0.999)

    def test_project_symmetric_matrix_to_psd(self):
        projected = _project_symmetric_matrix_to_psd(np.array([[1.0, 2.0],
                                                               [2.0, 1.0]]))
        self.assert_close_arrays(projected, [[1.5, 1.5], [1.5, 1.5]])
        positive = np.array([[2.0, 0.5], [0.5, 1.0]])
        self.assert_close_arrays(_project_symmetric_matrix_to_psd(positive), positive)


class TestLogPosteriorPredictive(MatcalUnitTest):
    def setUp(self):
        super().setUp(__file__)
        self.sensitivities = np.array([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]])
        self.residuals = np.array([[0.1, -0.2, 0.05], [-0.1, 0.3, 0.0]])
        self.theta = _to_theta(np.array([0.5, 0.2]), np.array([0.1]))

    def test_log_determinant_of_positive_definite_matrix(self):
        log_determinant = _get_log_determinant_of_positive_definite_matrix(
            np.eye(2) * np.e)
        self.assertAlmostEqual(log_determinant, 2.0)
        self.assertIsNone(_get_log_determinant_of_positive_definite_matrix(
            np.array([[1.0, 2.0], [2.0, 1.0]])))
        self.assertIsNone(_get_log_determinant_of_positive_definite_matrix(
            np.zeros((2, 2))))

    def test_inverse_methods_agree_and_are_finite(self):
        values = [_log_posterior_predictive(self.theta, self.sensitivities,
                                            self.residuals, 0.01, method)
                  for method in ["pseudoinverse", "solve"]]
        self.assertTrue(np.all(np.isfinite(values)))
        self.assertAlmostEqual(values[0], values[1])
        diagonal = _log_posterior_predictive(self.theta, self.sensitivities,
                                             self.residuals, 0.0, "diagonal")
        self.assertTrue(np.isfinite(diagonal))

    def test_not_positive_definite_candidate_is_rejected(self):
        with patch("matcal.core.parameter_studies."
                   "_get_log_determinant_of_positive_definite_matrix",
                   return_value=None):
            value = _log_posterior_predictive(self.theta, self.sensitivities,
                                              self.residuals, 0.01)
        self.assertEqual(value, -np.inf)
