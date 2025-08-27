from collections import OrderedDict
from copy import deepcopy
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
                                           VoronoiTessellation,
                                           KFoldCrossValidation,
                                           LeaveOneOutCrossValidation,
                                           VoronoiBatchStudy, )
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
    return {"x": x, "y": y}


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
        results = study.launch(nsamples, skip=skip)
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
            
    def setup_study(self, parameter_collection, model, objective): 
        study = self._study_class(parameter_collection, scramble=False, rng=42)
        study.add_evaluation_set(model, objective)
        return study
    
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
        nsamples = 10
        model_name = 'oneD'
        par_names = ['theta']
        
        # set up model, objective, paramter collection
        model = PythonModel(oneD_model)
        model.set_name(model_name)
        parameter_collection = TestHaltonStudy.setup_1d_parameter_collection()
        test_points = np.linspace(-2, 2, 5) 
        objective = SimulationResultsSynchronizer("x", test_points, "y")

        # run initial study
        study = self.setup_study(parameter_collection, model, objective)
        params, sim_history, state0 = TestHaltonStudy.run_study(study, nsamples, model_name, par_names)
        print(params.shape)
        self.assertEqual(len(params), nsamples)
        
        data = [sim_history[i]['y'][0] for i in range(nsamples)]

        # evaluate model at test points
        test_points = np.linspace(-2, 2)
        test_data = []
        import time
        for val in test_points: 
            th = Parameter("theta", -2, 2, distribution="uniform_uncertain", current_value=val)
            pc = ParameterCollection("predictions", th)
            res = model.run(state0, pc)
            test_data.append(res.results_data['y'][0])
            time.sleep(0.1)
        test_data = np.array(test_data)
         
        # interpolate and calculate prediction error of test points
        pred_error, y_pred = TestHaltonStudy.calculate_interpolated_pred_error(\
            params, data, test_points, test_data)

        # continue study with additional Halton samples
        nnew_samples = 12
        study = self.setup_study(parameter_collection, model, objective)
        study.restart()
        new_params, sim_history, state0 = TestHaltonStudy.run_study(study, nsamples+nnew_samples, model_name, par_names)
        self.assertEqual(len(new_params), nsamples + nnew_samples)

        new_data = [sim_history[i]['y'][0] for i in range(nsamples+nnew_samples)]

        # interpolate and calculate prediction error of test points
        new_pred_error, y_pred = TestHaltonStudy.calculate_interpolated_pred_error(\
            new_params, new_data, test_points, test_data)
        
        # prediction error should be less with more Halton samples
        self.assertGreater(pred_error, new_pred_error)
    

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
        print(study_param_results["grad_key:a"])
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
       
        print("Avg. resids:", np.average(resids))
        resids = resids.T
        noise_guess = std**2
        print(noise_guess)
        sens = resid_sensitivity(mean_dict).T
        print("Avg. sens:", np.average(sens))
        cov_est = _estimate_parameter_covariance(resids, sens, noise_guess)
        print("Est covar", cov_est)
        start = np.copy(cov_est)
        print("Initial covar:", start)
        fitted_posterior = _fit_posterior(resids, sens, start, noise_guess, method=None)
        print("Fitted posterior:", fitted_posterior)
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


class TestVoronoiTessellation(MatcalUnitTest):
    from scipy.spatial import voronoi_plot_2d, ConvexHull
    import matplotlib.pyplot as plt
    from matplotlib.path import Path
    from matplotlib.patches import Polygon as MplPolygon
    
    @staticmethod
    def fun2D(x):
        return np.sin(np.sqrt(x[:, 0]**2 + x[:, 1]**2))

    @staticmethod
    def fun3D(x):
        pass
    
    @staticmethod
    def initialization_2d(nsamples, bounds, seed=20):
        from scipy.stats.qmc import Halton
        from scipy.stats import qmc

        model = TestVoronoiTessellation.fun2D
        dim = 2
                
        l_bounds = [bounds[i][0] for i in np.arange(dim)]
        u_bounds = [bounds[i][1] for i in np.arange(dim)]

        # Generate initial points from Halton Sequence
        sampler = Halton(d=dim, seed=seed)
        X_unscaled = sampler.random(n=nsamples)
        X_init = qmc.scale(X_unscaled, l_bounds, u_bounds)
        y_init = model(X_init)

        ntest_samples = 40
        test_pts_sampler = Halton(d=dim, seed=seed+2)
        X_test_unscaled = np.atleast_2d(test_pts_sampler.random(n=ntest_samples))
        X_test = np.atleast_2d(qmc.scale(X_test_unscaled, l_bounds, u_bounds))
        y_test = model(X_test)
        print(y_test.shape) 
        
        return X_init, y_init, X_test, y_test, bounds
    
    def setUp(self):
        super().setUp(__file__)

    def test_2d_initialization(self):
        nsamples = 4
        bounds = [[-5, 5], [-5, 5]]
        X_init, _, _, _, bounds = TestVoronoiTessellation.initialization_2d(nsamples, bounds)
        
        for fo in [True, False]:
            vor = VoronoiTessellation(X_init, bounds, finite_only=fo)
        
            # Validate that ghost points are created correctly and that _all_points
            # includes both original and ghost points.
            self.assertEqual(vor._ghost_points.shape, (8, 2))
            min_x, max_x = bounds[0]
            min_y, max_y = bounds[1]
            for point in vor._ghost_points:
                x, y = point
                self.assertTrue(x < min_x or x > max_x or y < min_y or y > max_y,
                                msg=f"Ghost point {point} is inside the bounding box.")
                
            nghost = vor._ghost_points.shape[0]
            self.assertEqual(vor._all_points.shape[0], nsamples + nghost, msg="vor._all_points does not have correct dimensions.")
            self.assertEqual(vor._all_points[:nsamples, :].tolist(), X_init.tolist(), msg="vor._all_points does not contain X_init")
            self.assertEqual(vor._all_points[nsamples:, :].tolist(), vor._ghost_points.tolist(), msg="vor._all_points does not contain ghost points.")
            
            self.assertTrue(all(vor._boo[nsamples:]), msg="Ghost points not correctly identified.")
            self.assertFalse(any(vor._boo[:nsamples]), msg="Ghost points not correctly identified") 
            
            # Check that all training points belong to regions that contain no infinite vertices
            training_point_regions = vor.vor.point_region[:nsamples].tolist()
            finite_training_regions = [i for i in training_point_regions if -1 not in vor.vor.regions[i]]
            self.assertEqual(training_point_regions, finite_training_regions)
            
            # Check that the dimension of the voronoi tessellation is correct
            self.assertEqual(2, vor.ndim, msg="Dimension of voronoi tessellation not correct.")
        
            # Check that ConvexHull is created only when finite_only is False
            if fo:
                self.assertIsNone(vor.boundary_hull, msg="Boundary hull created for finite_only=False.")

    def test_2d_identify_vertices_outside_bounds(self):
        nsamples = 4
        bounds = [[-5, 5], [-5, 5]]
        X_init, _, _, _, bounds = TestVoronoiTessellation.initialization_2d(nsamples, bounds)
        vor = VoronoiTessellation(X_init, bounds)
        min_x, max_x = bounds[0]
        min_y, max_y = bounds[1]
       
        for point_idx in np.arange(nsamples):
            region_idx = vor.get_voronoi_region(vor.vor.points[point_idx])[0][0]
            region = vor.vor.regions[region_idx]
            updated_region = vor.identify_vertices_outside_bounds(region)
            outside_vertices = [region[i] for i in np.arange(len(updated_region)) if updated_region[i] < 0]
            for vertex_idx in outside_vertices:
                vertex = vor.vor.vertices[vertex_idx]
                x, y = vertex
                self.assertTrue(x < min_x or x > max_x or y < min_y or y > max_y,
                                msg=f"Identified 'outside' vertex {vertex} is inside the bounding box.")
                vor_region = vor.get_voronoi_region(vertex)[0]
                self.assertIn(region_idx, vor_region, msg="identified vertex not in region")

    def test_2d_find_boundary_hull_ray_crossing(self):
        nsamples = 4
        bounds = [[-5, 5], [-5, 5]]
        X_init, _, _, _, bounds = TestVoronoiTessellation.initialization_2d(nsamples, bounds)
        vor = VoronoiTessellation(X_init, bounds)

        # test crossing        
        U = np.array([1, 1])  # Example ray direction
        z = np.array([0, 0])  # Example ray origin
        expected_result = np.array([5, 5])  # Replace with the expected intersection point
        result = vor.find_boundary_hull_ray_crossings(U, z)
        self.assertTrue(np.all(result == expected_result))

        # test no crossing
        z = np.array([6, 6])  # Origin above the convex hull
        U = np.array([1, 1])  # Direction that does not intersect
        result = vor.find_boundary_hull_ray_crossings(U, z)
        self.assertIsNone(result) 
    
    def test_2d_get_region_vertices(self):
        from scipy.spatial import voronoi_plot_2d, ConvexHull
        import matplotlib.pyplot as plt
        from matplotlib.path import Path
        from matplotlib.patches import Polygon as MplPolygon
        nsamples = 4
        bounds = [[-5, 5], [-5, 5]]
        X_init, _, _, _, bounds = TestVoronoiTessellation.initialization_2d(nsamples, bounds)
        vor = VoronoiTessellation(X_init, bounds)
        
        for pt_idx in np.arange(nsamples):
            region_idx = vor.get_voronoi_region(vor.vor.points[pt_idx])[0][0]
            region_vertices = vor.get_region_vertices(region_idx, identify_outside_vertices=False)
            bounded_region_vertices = vor.get_region_vertices(region_idx, identify_outside_vertices=True)

            _, ax = plt.subplots(figsize=(12,8))
            voronoi_plot_2d(vor.vor, ax=ax, show_vertices=True,
                            line_width=2)
            ax.plot(X_init[:, 0], X_init[:, 1], '.', markersize=10, color='m', label='Training Points')
            plt.legend(fontsize=20)
            #plt.savefig(f"/ascldap/users/dericci/voronoi_tessellation.png")

            ax.plot(region_vertices[:, 0], region_vertices[:, 1], '.', markersize=15, color='g', label='R1 Vertices')
            plt.legend(fontsize=20)
            plt.savefig(f"/ascldap/users/dericci/voronoi_tessellation_r{region_idx}_vertices.png")
            plt.close()
        
            _, ax = plt.subplots(figsize=(12,8))
            voronoi_plot_2d(vor.vor, ax=ax, show_vertices=True,
                            line_width=2)
            ax.plot(X_init[:, 0], X_init[:, 1], '.', markersize=10, color='m', label='Training Points')
            ax.plot(bounded_region_vertices[:, 0], bounded_region_vertices[:, 1], '.', color='r', markersize=15, label=f'R{region_idx} Bounded Vertices')
            for simplex in vor.boundary_hull.simplices:
                plt.plot(vor.boundary_points[simplex, 0], vor.boundary_points[simplex, 1], 'k-', lw=2)
            plt.legend(fontsize=20)
            plt.savefig(f"/ascldap/users/dericci/voronoi_tessellation_r{region_idx}_bounded_vertices.png")
            plt.close()

            self.assertEqual(vor._all_points.shape[0], len(vor.vor.point_region))

            # compare convex hulls of original and bounded region vertices
            # the area of the original hull should be >= the area of the bounded hull
            # the bounded hull should reside completely within the original hull
            region_hull = ConvexHull(region_vertices)
            region_hull_pts = region_vertices[region_hull.vertices]
            region_path = Path(region_hull_pts)

            bounded_region_hull = ConvexHull(bounded_region_vertices)
            bounded_hull_pts = bounded_region_vertices[bounded_region_hull.vertices]
            is_inside = region_path.contains_points(bounded_hull_pts, radius=1e-10)
            self.assertGreaterEqual(region_hull.area, bounded_region_hull.area)
            self.assertTrue(np.all(is_inside))
                
            # Plotting
            _, ax = plt.subplots(figsize=(12, 12))

            # Plot original points
            ax.plot(region_vertices[:, 0], region_vertices[:, 1], 'bo', label='Outer Points')
            ax.plot(bounded_region_vertices[:, 0], bounded_region_vertices[:, 1], 'ro', label='Inner Points')

            # Draw convex hulls as filled polygons
            outer_patch = MplPolygon(region_hull_pts, closed=True, fill=False, edgecolor='blue', linewidth=2, label='Outer Hull')
            inner_patch = MplPolygon(bounded_hull_pts, closed=True, fill=False, edgecolor='red', linewidth=2, label='Inner Hull')
            ax.add_patch(outer_patch)
            ax.add_patch(inner_patch)

            # Optional: mark inner hull vertices that are not contained
            for pt, inside in zip(bounded_hull_pts, is_inside):
                if not inside:
                    ax.plot(pt[0], pt[1], 'kx', markersize=10, label='Outside Hull Vertex')

            ax.legend()
            ax.set_title('Convex Hull Containment Test')
            ax.set_aspect('equal')
            plt.grid(True)
            plt.savefig(f"/ascldap/users/dericci/inner_outer_hull_r{region_idx}.png")
            plt.close("all")       
         
    def test_2d_get_voronoi_vertices(self):
        from matplotlib.path import Path
        nsamples = 4
        bounds = [[-5, 5], [-5, 5]]
        X_init, _, _, _, bounds = TestVoronoiTessellation.initialization_2d(nsamples, bounds)
        vor = VoronoiTessellation(X_init, bounds)
        boundary_hull = vor.boundary_hull
        boundary_hull_points = boundary_hull.vertices
        boundary_path = Path(vor.boundary_points[boundary_hull_points])
        region_vertices = np.empty((0, 2))
                
        # voronoi vertices of all training points
        vor_vertices = vor.get_voronoi_vertices(identify_outside_vertices=False)

        # check that the vertices returned by vor.get_voronoi_vertices are the same as
        # the region vertices returned by vor.get_region_vertices for all regions
        for pt_idx in np.arange(nsamples):
            region_index = vor.get_voronoi_region(vor.vor.points[pt_idx])[0][0]
            region_vertices = np.vstack([region_vertices, vor.get_region_vertices(region_index, identify_outside_vertices=False)])
        unique_vertices = set(tuple(row) for row in region_vertices)
        vertices = np.asarray([list(row) for row in unique_vertices])
        self.assertEqual(set(map(tuple, vor_vertices)), set(map(tuple, vertices)))
        
        # voronoi vertices snipped to boundary - check that all bounded vertices are within Convex
        # hull defined by boundary points
        bounded_vor_vertices = vor.get_voronoi_vertices(identify_outside_vertices=True)
        is_inside = boundary_path.contains_points(bounded_vor_vertices, radius=1e-10)
        self.assertTrue(np.all(is_inside))

        # check that the bounded vertices returned by vor.get_voronoi_vertices are the same as
        # the bounded region vertices returned by vor.get_region_vertices for all regions
        region_vertices = np.empty((0, 2))
        for pt_idx in np.arange(nsamples):
            region_index = vor.get_voronoi_region(vor.vor.points[pt_idx])[0][0]
            region_vertices = np.vstack([region_vertices, vor.get_region_vertices(region_index, identify_outside_vertices=True)])
        unique_vertices = set(tuple(row) for row in region_vertices)
        vertices = np.asarray([list(row) for row in unique_vertices])
        self.assertEqual(set(map(tuple, bounded_vor_vertices)), set(map(tuple, vertices)))
        
    def test_2d_get_closest_seed(self):
        nsamples = 4
        bounds = [[-5, 5], [-5, 5]]
        X_init, _, _, _, bounds = TestVoronoiTessellation.initialization_2d(nsamples, bounds)
        vor = VoronoiTessellation(X_init, bounds)
        
        # point close to seed: should return seed
        for i in np.arange(nsamples):
            test_point = vor.points[i] * 1.01
            closest_point = vor.vor.points[vor.get_closest_seed(test_point)]
            self.assertTrue(np.all(closest_point == vor.points[i]))
            
        # vertices of seed region: should return multiple seeds, including given seed
        for pt_idx in np.arange(nsamples):
            seed = vor.points[pt_idx]
            region_index = vor.get_voronoi_region(seed)[0][0]
            region_vertices = vor.vor.vertices[vor.vor.regions[region_index]]
            for vertice in region_vertices:
                closest_point_indices = vor.get_closest_seed(vertice)
                self.assertGreater(len(closest_point_indices), 1)
                closest_points = vor.vor.points[closest_point_indices]
                self.assertTrue(seed in closest_points)
    
    def test_2d_get_voronoi_region(self):
        from matplotlib.patches import Polygon
        from matplotlib.path import Path
        import random
        
        # create polygon from region vertices
        # sample point from within polygon, and assert that point is in the region
        
        nsamples = 4
        bounds = [[-5, 5], [-5, 5]]
        X_init, _, _, _, bounds = TestVoronoiTessellation.initialization_2d(nsamples, bounds)
        vor = VoronoiTessellation(X_init, bounds)

        # loop through all regions
        for region_idx, region in enumerate(vor.vor.regions):
           
            if -1 in region: # skip over regions with infinite vertices (all seeds have finite vertices)
                continue
            if not region: # skip over empty regions
                continue
             
            # get region vertices
            vertices = vor.vor.vertices[region]
            
            # create polygon
            polygon = Polygon(vertices, closed=True)
            path = polygon.get_path()
            
            # sample a point
            max_attempts = 1000000
            minx, miny = polygon.get_extents().min
            maxx, maxy = polygon.get_extents().max
            point_found = False
            for iter in range(max_attempts):
                # random point within bounding box of polygon
                point = (random.uniform(minx, maxx), random.uniform(miny, maxy))
                 # check if point is inside polygon
                if path.contains_point(point):
                    point_found = True
                    break
                else:
                    continue
            
            if not point_found:
                print(f"failed to sample point inside polygon {region_idx} after many attempts.")
                continue
        
            # check that get_voronoi_region returns given region
            voronoi_region = vor.get_voronoi_region(point)
            self.assertTrue(region_idx == voronoi_region[0][0])
               
    def test_2d_get_region_seed(self):
        nsamples = 4
        bounds = [[-5, 5], [-5, 5]]
        X_init, _, _, _, bounds = TestVoronoiTessellation.initialization_2d(nsamples, bounds)
        vor = VoronoiTessellation(X_init, bounds)
        for pt_idx in np.arange(nsamples):
            region_index = vor.get_voronoi_region(vor.vor.points[pt_idx])[0][0]
            seed = vor.get_region_seed(region_index)
            region_point_idx, = np.where(vor.vor.point_region == region_index)
            region_seed = vor.points[region_point_idx]
            self.assertTrue(np.all(seed == region_seed))
    
    def test_2d_find_furthest_vertex(self):
        nsamples = 4
        bounds = [[-5, 5], [-5, 5]]
        X_init, _, _, _, bounds = TestVoronoiTessellation.initialization_2d(nsamples, bounds)
        vor = VoronoiTessellation(X_init, bounds)
        
        for pt_idx in np.arange(nsamples):
            seed = vor.points[pt_idx]
            region_index = vor.get_voronoi_region(seed)[0][0]
            region = vor.vor.regions[region_index]

            # without snipping vertices: assert the identified furthest vertex has the greatest distance
            vertices = vor.get_region_vertices(region_index, identify_outside_vertices=False)
            all_vertices, furthest_vertex = vor.find_furthest_vertex(region_index, identify_outside_vertices=False)
            distances = np.linalg.norm(seed - all_vertices, axis=1)
            max_dist = np.argmax(distances)
            self.assertEqual(max_dist, furthest_vertex)
            self.assertTrue(np.all(all_vertices == vertices))
            
            # with snipping vertices: assert the identified furthest vertex has the greatest distance
            vertices = vor.get_region_vertices(region_index, identify_outside_vertices=True)
            all_vertices, furthest_vertex = vor.find_furthest_vertex(region_index, identify_outside_vertices=True)
            distances = np.linalg.norm(seed - all_vertices, axis=1)
            max_dist = np.argmax(distances)
            self.assertEqual(max_dist, furthest_vertex)
            self.assertTrue(np.all(all_vertices == vertices))


class TestKFoldCrossValidation(MatcalUnitTest):
    
    def setUp(self):
        super().setUp(__file__)

        from sklearn.linear_model import LinearRegression
        # Sample data for testing
        self.X = np.array([[1], [2], [3], [4], [5]])
        self.y = np.array([1, 2, 3, 4, 5])
        self.nsamples = len(self.X)
        self.model = LinearRegression()
        self.kfold = KFoldCrossValidation(model=self.model, n_splits=5)

    def test_initialization(self):
        from sklearn.linear_model import LinearRegression
        self.assertEqual(self.kfold.n_splits, 5)
        self.assertIsInstance(self.kfold.model, LinearRegression)
        self.assertFalse(self.kfold.group_kfold)
        self.assertIsNone(self.kfold.scale)

    def test_calculate_sum_abs_error(self):
        y_true = np.array([1, 2, 3])
        y_pred = np.array([1, 2, 4])
        error = self.kfold.calculate_sum_abs_error(y_true, y_pred)
        self.assertEqual(error, 1)

    def test_calculate_mean_abs_perc_error(self):
        y_true = np.array([1, 2, 3])
        y_pred = np.array([1, 2, 4])
        error = self.kfold.calculate_mean_abs_perc_error(y_true, y_pred)
        self.assertAlmostEqual(error, 11.11, places=2)

    def test_calculate_sum_abs_perc_error(self):
        y_true = np.array([1, 2, 3])
        y_pred = np.array([1, 2, 4])
        error = self.kfold.calculate_sum_abs_perc_error(y_true, y_pred)
        self.assertAlmostEqual(error, 33.33, places=2)

    def test_calculate_mse(self):
        y_true = np.array([1, 2, 3])
        y_pred = np.array([1, 2, 4])
        error = self.kfold.calculate_mse(y_true, y_pred)
        self.assertAlmostEqual(error, 0.33, places=2)

    def test_calculate_rmse(self):
        y_true = np.array([1, 2, 3])
        y_pred = np.array([1, 2, 4])
        error = self.kfold.calculate_rmse(y_true, y_pred)
        self.assertAlmostEqual(error, 0.57735, places=5)

    def test_perform_kfold_cv(self):
        metric = 'mse'
        kf_results = self.kfold.perform_kfold_cv(self.X, self.y, metric, groups=None)
       
        # Check that the results are in the expected format
        self.assertIsInstance(kf_results, dict)
        self.assertEqual(len(kf_results), self.kfold.n_splits)

        # Check that each result is a tuple (kfold error, indices of test samples)
        test_indices = []
        for result in kf_results.values():
            self.assertIsInstance(result, tuple)
            self.assertEqual(len(result), 2)
            self.assertIsInstance(result[0], float)  # error
            self.assertIsInstance(result[1], np.ndarray)  # test indices
            test_indices.append(result[1])
        test_indices = np.vstack(test_indices)
        
        # check that each test sample is used once and only once
        self.assertTrue(np.all(np.isin(np.arange(self.nsamples), test_indices)))
        self.assertEqual(len(np.unique(test_indices)), self.nsamples)
            
    def test_cross_val_fold(self):
        train_index = [0, 1, 2]
        test_index = [3, 4]
        error, test_idx_returned = self.kfold.cross_val_fold(train_index, test_index, self.X, self.y, 'mse')
        self.assertEqual(test_idx_returned, test_index)
        self.assertIsInstance(error, float)          
       
        
class TestLeaveOneOutCrossValidation(MatcalUnitTest):
    
    def setUp(self):
        super().setUp(__file__)

        from sklearn.linear_model import LinearRegression
        # Sample data for testing
        self.X = np.array([[1], [2], [3], [4], [5]])
        self.y = np.array([1, 2, 3, 4, 5])
        self.nsamples = len(self.X)
        self.model = LinearRegression()
        self.loocv = LeaveOneOutCrossValidation(model=self.model)

    def test_initialization(self):
        from sklearn.linear_model import LinearRegression
        self.assertIsInstance(self.loocv.model, LinearRegression)
        self.assertIsNone(self.loocv.scale)

    def test_calculate_sum_abs_error(self):
        y_true = np.array([1, 2, 3])
        y_pred = np.array([1, 2, 4])
        error = self.loocv.calculate_sum_abs_error(y_true, y_pred)
        self.assertEqual(error, 1)

    def test_calculate_mean_abs_perc_error(self):
        y_true = np.array([1, 2, 3])
        y_pred = np.array([1, 2, 4])
        error = self.loocv.calculate_mean_abs_perc_error(y_true, y_pred)
        self.assertAlmostEqual(error, 11.11, places=2)

    def test_calculate_mse(self):
        y_true = np.array([1, 2, 3])
        y_pred = np.array([1, 2, 4])
        mse = self.loocv.calculate_mse(y_true, y_pred)
        self.assertAlmostEqual(mse, 0.33, places=2)

    def test_calculate_rmse(self):
        y_true = np.array([1, 2, 3])
        y_pred = np.array([1, 2, 4])
        rmse = self.loocv.calculate_rmse(y_true, y_pred)
        self.assertAlmostEqual(rmse, 0.57735, places=5)

    def test_loo_val(self):
        metrics = ['sum_abs_error', 'mape', 'mse', 'rmse', 'sum_abs_perc_error']
        for metric in metrics:
            for i in np.arange(self.nsamples):
                error, index = self.loocv.loo_val(self.X, self.y, metric, i)
                self.assertEqual(index, i)
                self.assertIsInstance(error, float)
        
    def test_perform_loocv(self):
        indices = range(self.nsamples)
        metrics = ['sum_abs_error', 'mape', 'mse', 'rmse', 'sum_abs_perc_error']
        for metric in metrics:
            loo_results = self.loocv.perform_loocv(self.X, self.y, indices, metric=metric)
            
            self.assertIsInstance(loo_results, dict)
            self.assertEqual(len(loo_results), self.nsamples)
            self.assertIn(0, loo_results)  # Check if the first index is present in results
            
            # Check that each result is a tuple (r, indices of test samples)
            test_indices = []
            for result in loo_results.values():
                self.assertIsInstance(result, tuple)
                self.assertEqual(len(result), 2)
                self.assertIsInstance(result[0], float)  # error
                self.assertIsInstance(result[1], int)  # test indices
                test_indices.append(result[1])
            test_indices = np.vstack(test_indices)
            
            # check that each test sample is used once and only once
            self.assertTrue(np.all(np.isin(np.arange(self.nsamples), test_indices)))
            self.assertEqual(len(np.unique(test_indices)), self.nsamples)


class SurrogateModel:
    def __init__(self, input_scaling=False, output_scaling=True, **kwargs):
        self.input_scaling = input_scaling
        self.surrogate_type = 'gaussian_process'

        self.nrestarts = 50
        self.alpha = 1.0e-8
        self.normalize_y = output_scaling
        self.random_state = None
        self.output_scaler_with_std = True
        for kwarg in kwargs:
            if kwarg == 'n_restarts_optimizer':
                self.nrestarts = kwargs[kwarg]
            if kwarg == 'alpha':
                self.alpha = kwargs[kwarg]
            if kwarg == 'normalize_y':
                self.normalize_y = kwargs[kwarg]
            if kwarg == 'random_state':
                self.random_state = kwargs[kwarg]
            if kwarg == 'output_scaler_with_std':
                self.output_scaler_with_std = kwargs[kwarg]

    def fit(self, X, y):
        from sklearn.preprocessing import StandardScaler
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

        X = np.atleast_2d(X)
        self.nsamples = X.shape[0]
        self.nfeatures = X.shape[1]

        gp_train_features = X
        if self.input_scaling:
            self.input_scaler = StandardScaler()
            scaled_features = self.input_scaler.fit_transform(X)
            gp_train_features = scaled_features
        gp_train_targets = y

        # squared exponential kernel
        self.kernel = C(1.0, constant_value_bounds=(1e-3, 1e3))\
            * RBF(np.ones(self.nfeatures), length_scale_bounds=(1e-3, 1e3))

        self.surrogate = \
            GaussianProcessRegressor(
                kernel=self.kernel,
                n_restarts_optimizer=self.nrestarts,
                alpha=self.alpha,
                random_state=self.random_state,
                normalize_y=self.normalize_y)
        self.surrogate.fit(
            gp_train_features,
                gp_train_targets)


    def predict(self, X):

        X = np.atleast_2d(X)
        assert X.ndim == 2
        assert X.shape[1] == self.nfeatures

        if self.input_scaling:
            X = self.input_scaler.transform(X)

        return self.surrogate.predict(X)
   
class TestVoronoiBatchStudy(MatcalUnitTest):
    @staticmethod
    def fun2D(x):
        return np.sin(np.sqrt(x[:, 0]**2 + x[:, 1]**2))

    @staticmethod
    def fun3D(x):
        pass
    
    @staticmethod
    def initialization_2d(nsamples, bounds, seed=20):
        from scipy.stats.qmc import Halton
        from scipy.stats import qmc

        model = TestVoronoiBatchStudy.fun2D
        dim = 2
                
        l_bounds = [bounds[i][0] for i in np.arange(dim)]
        u_bounds = [bounds[i][1] for i in np.arange(dim)]
        dim = len(bounds)
        
        # Generate initial points from Halton Sequence
        sampler = Halton(d=dim, seed=seed)
        X_unscaled = sampler.random(n=nsamples)
        X_init = qmc.scale(X_unscaled, l_bounds, u_bounds)
        y_init = model(X_init)

        ntest_samples = 40
        test_pts_sampler = Halton(d=dim, seed=seed+2)
        X_test_unscaled = np.atleast_2d(test_pts_sampler.random(n=ntest_samples))
        X_test = np.atleast_2d(qmc.scale(X_test_unscaled, l_bounds, u_bounds))
        y_test = model(X_test)
        
        # Initialize surrogate model with initial training data
        surr_model = SurrogateModel(input_scaling=True)
        surr_model.fit(X_init, y_init)
        return X_init, y_init, X_test, y_test, surr_model
    
    def setUp(self):
        super().setUp(__file__)
        
    def test_initialization(self):
        nsamples = 4
        bounds = [[-5, 5], [-5, 5]]
        X_init, y_init, X_test, y_test, surr_model = TestVoronoiBatchStudy.initialization_2d(nsamples, bounds)
        vor_study = VoronoiBatchStudy(surr_model, bounds, X_test, y_test, rng=42)
        self.assertFalse(vor_study.finite_only)
        self.assertTrue(vor_study.iterative_updates)
        self.assertEqual(vor_study.surr_model, surr_model)
        self.assertEqual(vor_study.bounds, bounds)
        self.assertTrue(np.all(vor_study.X_test == X_test))
        self.assertTrue(np.all(vor_study.y_test == y_test))
        self.assertEqual(vor_study.surr_model_type, 'GPR')
        self.assertEqual(vor_study.voronoi_type, 'full')
        
        expected_boundary_points = np.array([[-5, -5],[5, -5],[-5, 5],[5, 5]])
        self.assertTrue(np.all(vor_study.boundary_points == expected_boundary_points))
    
    def test_calculate_errors(self):
        pass
    
    def test_surrogate_loss(self):
        pass
    
    def test_perform_voronoi_batch_sampling(self):
        pass
    
    def test_launch(self):
        # test last after other attributes tested
        pass 
    def test_placeholder(self):
        if True:
            plt.close("all")
            X_df = pd.DataFrame(X)
            X_df['label'] = 'Training'
            test_df = pd.DataFrame(X_test)
            test_df['label'] = 'Test'
            data = pd.concat([X_df, test_df])
            palette = {'Training': 'blue', 'Test': 'red'}
            sns.set_context("paper", rc={"xlabel.fontsize": 16, "ylabel.fontsize": 16,\
                "xlabel.fontweight": "bold", "ylabel.fontweight": "bold"})
            pairplot = sns.pairplot(data, hue='label', palette=palette, corner=True,
                plot_kws=dict(marker='.', s=20))
            pairplot.fig.set_size_inches(5, 5)
            plt.savefig(f"{figpath}/training_points_iter_{batch_number}.png")
            plt.close()

        if dim == 2 and voronoi_type == 'full':
            fix, ax = plt.subplots()
            voronoi_plot_2d(voronoi_tessellation.vor, ax=ax, show_vertices=False,
                line_width=2)
            ax.plot(X[:, 0], X[:, 1], '.', markersize=10, color='m', label='Training Points')
            plt.legend(fontsize=20)
            plt.savefig(f"{figpath}/voronoi_tessellation_iter_{batch_number}.png")
            plt.close()

            fig, ax = plt.subplots(figsize=(12,8))
            ax.plot(nsamples_list, voronoi_pred_error, linestyle='--', marker='o',  markersize=10, color='fuchsia', label='Voronio')
            plt.legend(fontsize=20)
            plt.xticks(np.arange(nsamples_list[0], nsamples_list[-1], 5), fontsize=16)
            plt.xlabel('Number of Samples', fontsize=20)
            plt.ylabel('MSE', fontsize=20)
            plt.yscale('log')
            plt.title('Surrogate Prediction Error', fontsize=20)
            plt.savefig(f'{figpath}/prediction_error.png')

            plt.close("all")

    