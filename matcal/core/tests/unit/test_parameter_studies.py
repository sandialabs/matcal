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
                                           VoronoiAdaptiveSurrogateStudy, )
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

def quadratic_model_2d(**parameters):
    """Quadratic curve: f = a + bx. x independent variable."""
    x = np.linspace(0, 1, 100)
    a = parameters['a']
    b = parameters['b']
    f = a + b * x
    return {"x":x, "f":f}

def quadratic_model_3d(**parameters):
    """Quadratic curve: f = a + bx + cx^2. x independent variable."""
    
    x = np.linspace(0, 1, 100)
    a = parameters['a']
    b = parameters['b']
    c = parameters['c']
    f = a + b * x + c * x ** 2
    return {"x":x, "f":f} 
    
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
            
    def setup_study(self, scramble=False): 
        self.model_name = 'oneD'
        self.par_names = ['theta']
        
        # set up model, objective, parameter collection
        self.model = PythonModel(oneD_model)
        self.model.set_name(self.model_name)
        parameter_collection = TestHaltonStudy.setup_1d_parameter_collection()
        test_points = np.linspace(-2, 2, 5) 
        objective = SimulationResultsSynchronizer("x", test_points, "f")
        self.study = self._study_class(parameter_collection, scramble=scramble, rng=42)
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
        print(params.shape)
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
        
        with self.assertRaises(self.study.StudyInputError):
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
    def funND(x):
        # Tang function
        d = x.shape[1]
        sum_ = 0
        for ii in np.arange(d):
            sum_ += (x[:, ii] ** 4) - (16 * x[:, ii] ** 2) + (5 * x[:, ii])
        quadrant_filter = np.all(x < 0, axis=1)
        sum_[quadrant_filter] = 0
        return 0.5 * sum_  
    
    @staticmethod
    def voronoi_initialization(dim, nsamples, bounds, seed=20):
        from scipy.stats.qmc import Halton
        from scipy.stats import qmc

        if dim == 2:
            model = TestVoronoiTessellation.fun2D
        elif dim == 3:
            model = TestVoronoiTessellation.funND
                
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

    @staticmethod
    def check_arrays_contain_same_elements(arr1, arr2):
        # check that initial_points are the same as the current points
        same_elements = np.all(np.isin(arr2, arr1))
        same_length = np.all(arr1.shape == arr2.shape)
        return same_elements, same_length
        
    def setUp(self):
        super().setUp(__file__)

    def test_initialization(self):
        dims = [2, 3]
        for dim in dims:
            nsamples = 2 ** dim
            bounds = [[-5, 5] for d in np.arange(dim)]
            X_init, _, _, _, bounds = TestVoronoiTessellation.voronoi_initialization(dim, nsamples, bounds)
            
            for fo in [True, False]:
                opts = {'finite_only':fo}
                vor = VoronoiTessellation(X_init, bounds)
                vor.build(**opts)
            
                # Validate that ghost points are created correctly and that _all_points
                # includes both original and ghost points.
                self.assertEqual(vor._ghost_points.shape[1], dim)
                min_x, max_x = bounds[0]
                min_y, max_y = bounds[1]
                if dim == 3:
                    min_z, max_z = bounds[2]
                for point in vor._ghost_points:
                    if dim == 2:
                        x, y = point
                        self.assertTrue(x < min_x or x > max_x or y < min_y or y > max_y,
                                        msg=f"Ghost point {point} is inside the bounding box.")
                    elif dim == 3:
                        x, y, z = point
                        self.assertTrue(x < min_x or x > max_x or y < min_y or y > max_y or z < min_z or z > max_z,
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
                self.assertEqual(dim, vor.ndim, msg="Dimension of voronoi tessellation not correct.")
            
                # Check that ConvexHull is created only when finite_only is False
                if fo:
                    self.assertIsNone(vor.boundary_hull, msg="Boundary hull created for finite_only=False.")

    def test_attribute_error(self):
        dim = 2
        nsamples = 2 ** dim
        bounds = [[-5, 5] for d in np.arange(dim)]
        X_init, _, _, _, bounds = TestVoronoiTessellation.voronoi_initialization(dim, nsamples, bounds)
        
        opts = {'_finite_only':True}
        vor = VoronoiTessellation(X_init, bounds)
        with self.assertRaises(AttributeError):
            vor.build(**opts)
        
    def test_identify_vertices_outside_bounds(self):
        dims = [2, 3]
        for dim in dims:
            nsamples = 2 ** dim
            bounds = [[-5, 5] for d in np.arange(dim)]
            X_init, _, _, _, bounds = TestVoronoiTessellation.voronoi_initialization(dim, nsamples, bounds)
            vor = VoronoiTessellation(X_init, bounds)
            vor.build()
            min_x, max_x = bounds[0]
            min_y, max_y = bounds[1]
            if dim == 3:
                min_z, max_z = bounds[2]
        
            for point_idx in np.arange(nsamples):
                region_idx = vor.get_voronoi_region(vor.vor.points[point_idx])[0][0]
                region = vor.vor.regions[region_idx]
                updated_region = vor.identify_vertices_outside_bounds(region)
                outside_vertices = [region[i] for i in np.arange(len(updated_region)) if updated_region[i] < 0]
                for vertex_idx in outside_vertices:
                    vertex = vor.vor.vertices[vertex_idx]
                    if dim == 2:
                        x, y = vertex
                        self.assertTrue(x < min_x or x > max_x or y < min_y or y > max_y,
                                        msg=f"Identified 'outside' vertex {vertex} is inside the bounding box.")
                    elif dim == 3:
                        x, y, z = vertex
                        self.assertTrue(x < min_x or x > max_x or y < min_y or y > max_y or z < min_z or z > max_z,
                                        msg=f"Identified 'outside' vertex {vertex} is inside the bounding box.")
                    vor_region = vor.get_voronoi_region(vertex)[0]
                    self.assertIn(region_idx, vor_region, msg="identified vertex not in region")

    def test_2d_find_boundary_hull_ray_crossing(self):
        nsamples = 4
        dim = 2
        bounds = [[-5, 5], [-5, 5]]
        X_init, _, _, _, bounds = TestVoronoiTessellation.voronoi_initialization(dim, nsamples, bounds)
        vor = VoronoiTessellation(X_init, bounds)
        vor.build()

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
    
    def test_3d_find_boundary_hull_ray_crossing(self):
        nsamples = 8
        dim = 3
        bounds = [[-5, 5], [-5, 5], [-5, 5]]
        X_init, _, _, _, bounds = TestVoronoiTessellation.voronoi_initialization(dim, nsamples, bounds)
        vor = VoronoiTessellation(X_init, bounds)
        vor.build()

        # test crossing        
        U = np.array([1, 1, 1])  # Example ray direction
        z = np.array([0, 0, 0])  # Example ray origin
        expected_result = np.array([5, 5, 5])  # Replace with the expected intersection point
        result = vor.find_boundary_hull_ray_crossings(U, z)
        self.assertTrue(np.all(result == expected_result))

        # test no crossing
        z = np.array([6, 6, 6])  # Origin above the convex hull
        U = np.array([1, 1, 1])  # Direction that does not intersect
        result = vor.find_boundary_hull_ray_crossings(U, z)
        self.assertIsNone(result)
        
        # test two crossings
        z = np.array([-6, -6, -6]) # origin 
        U = np.array([1, 1, 1,]) # Direction
        result = vor.find_boundary_hull_ray_crossings(U, z)
        
    def test_get_region_vertices(self):
        # implicitly tests vor.replace_unbounded_vertices and vor.snip_ridge_vertices
        
        from scipy.spatial import voronoi_plot_2d, ConvexHull, Delaunay
        import matplotlib.pyplot as plt
        from matplotlib.path import Path
        from matplotlib.patches import Polygon as MplPolygon
        dims = [2, 3]
        for dim in dims:
            nsamples = 2 ** dim
            bounds = [[-5, 5] for d in np.arange(dim)]
            X_init, _, _, _, bounds = TestVoronoiTessellation.voronoi_initialization(dim, nsamples, bounds)
            for fo in [True, False]:
                opts = {'finite_only':fo}
                vor = VoronoiTessellation(X_init, bounds)
                vor.build(**opts)
            
                for pt_idx in np.arange(nsamples):
                    region_idx = vor.get_voronoi_region(vor.vor.points[pt_idx])[0][0]
                    region_vertices = vor.get_region_vertices(region_idx, identify_outside_vertices=False)
                    bounded_region_vertices = vor.get_region_vertices(region_idx, identify_outside_vertices=True)
                    for vert in region_vertices:
                        self.assertIn(region_idx, vor.get_voronoi_region(vert)[0]) # confirm vertices are within given region
                    if bounded_region_vertices is not None:
                        for vert in bounded_region_vertices:
                            self.assertIn(region_idx, vor.get_voronoi_region(vert)[0]) # confirm vertices are within given region
                            if not fo:
                                self.assertGreaterEqual(vor.bhullD.find_simplex(vert), 0) # confirm point is within boundary hull

                    self.assertEqual(vor._all_points.shape[0], len(vor.vor.point_region))

                    # compare convex hulls of original and bounded region vertices
                    # the area of the original hull should be >= the area of the bounded hull
                    # the bounded hull should reside completely within the original hull
                    if region_vertices.shape[0] > 2:
                        region_hull = ConvexHull(region_vertices)
                        region_hull_pts = region_vertices[region_hull.vertices]
                        delaunay_region = Delaunay(region_hull_pts) #Delaunay triangulation of region vertices
                    if bounded_region_vertices is not None and bounded_region_vertices.shape[0] > dim + 1:
                        try:
                            bounded_region_hull = ConvexHull(bounded_region_vertices)
                            bounded_hull_pts = bounded_region_vertices[bounded_region_hull.vertices]
                        except:
                            import pdb
                            pdb.set_trace()
                        # check if all points of the bounded hull are inside the original hulls
                        is_inside = delaunay_region.find_simplex(bounded_hull_pts) >= 0
                    
                        self.assertGreaterEqual(region_hull.area, bounded_region_hull.area)
                        self.assertTrue(np.all(is_inside))
                        
                    # 2d plots
                    if dim == 2:
                        _, ax = plt.subplots(figsize=(12,8))
                        voronoi_plot_2d(vor.vor, ax=ax, show_vertices=True,
                                        line_width=2)
                        ax.plot(X_init[:, 0], X_init[:, 1], '.', markersize=10, color='m', label='Training Points')
                        plt.legend(fontsize=20)
                        #plt.savefig(f"/ascldap/users/dericci/voronoi_tessellation.png")

                        ax.plot(region_vertices[:, 0], region_vertices[:, 1], '.', markersize=15, color='g', label='R1 Vertices')
                        plt.legend(fontsize=20)
                        plt.savefig(f"voronoi_tessellation_r{region_idx}_vertices.png")
                        plt.close()
                    
                        _, ax = plt.subplots(figsize=(12,8))
                        voronoi_plot_2d(vor.vor, ax=ax, show_vertices=True,
                                        line_width=2)
                        ax.plot(X_init[:, 0], X_init[:, 1], '.', markersize=10, color='m', label='Training Points')
                        if bounded_region_vertices is not None:
                            ax.plot(bounded_region_vertices[:, 0], bounded_region_vertices[:, 1], '.', color='r', markersize=15, label=f'R{region_idx} Bounded Vertices')
                        if not fo:
                            for simplex in vor.boundary_hull.simplices:
                                plt.plot(vor.boundary_points[simplex, 0], vor.boundary_points[simplex, 1], 'k-', lw=2)
                            plt.legend(fontsize=20)
                            plt.savefig(f"voronoi_tessellation_r{region_idx}_bounded_vertices.png")
                            plt.close()

                        # Fig of voronoi tessellation with boundary hull
                        _, ax = plt.subplots(figsize=(12, 12))

                        # Plot original points
                        ax.plot(region_vertices[:, 0], region_vertices[:, 1], 'bo', label='Outer Points')
                        if bounded_region_vertices is not None:
                            ax.plot(bounded_region_vertices[:, 0], bounded_region_vertices[:, 1], 'ro', label='Inner Points')

                        # Draw convex hulls as filled polygons
                        try:
                            outer_patch = MplPolygon(region_hull_pts, closed=True, fill=False, edgecolor='blue', linewidth=2, label='Outer Hull')
                            inner_patch = MplPolygon(bounded_hull_pts, closed=True, fill=False, edgecolor='red', linewidth=2, label='Inner Hull')
                            ax.add_patch(outer_patch)
                            ax.add_patch(inner_patch)

                            # Optional: mark inner hull vertices that are not contained
                            for pt, inside in zip(bounded_hull_pts, is_inside):
                                if not inside:
                                    ax.plot(pt[0], pt[1], 'kx', markersize=10, label='Outside Hull Vertex')

                        except:
                            continue

                        ax.legend()
                        ax.set_title('Convex Hull Containment Test')
                        ax.set_aspect('equal')
                        plt.grid(True)
                        plt.savefig(f"inner_outer_hull_r{region_idx}.png")
                        plt.close("all")       
         
    def test_get_voronoi_vertices(self):
        from matplotlib.path import Path
        from scipy.spatial import Delaunay
        dims = [2, 3]
        for dim in dims:
            nsamples = 2 ** dim
            bounds = [[-5, 5] for d in np.arange(dim)]
            X_init, _, _, _, bounds = TestVoronoiTessellation.voronoi_initialization(dim, nsamples, bounds)
            vor = VoronoiTessellation(X_init, bounds)
            vor.build()
            boundary_hull = vor.boundary_hull
            boundary_hull_points = vor.boundary_points[boundary_hull.vertices]
            #boundary_path = Path(vor.boundary_points[boundary_hull_points]) # only works for 2D
            delaunay_region = Delaunay(boundary_hull_points) #Delaunay triangulation of region vertices
            region_vertices = np.empty((0, dim))
                    
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
            #is_inside = boundary_path.contains_points(bounded_vor_vertices, radius=1e-10)
            is_inside = delaunay_region.find_simplex(bounded_vor_vertices) >= 0
            self.assertTrue(np.all(is_inside))

            # check that the bounded vertices returned by vor.get_voronoi_vertices are the same as
            # the bounded region vertices returned by vor.get_region_vertices for all regions
            region_vertices = np.empty((0, dim))
            for pt_idx in np.arange(nsamples):
                region_index = vor.get_voronoi_region(vor.vor.points[pt_idx])[0][0]
                region_vertices = np.vstack([region_vertices, vor.get_region_vertices(region_index, identify_outside_vertices=True)])
            unique_vertices = set(tuple(row) for row in region_vertices)
            vertices = np.asarray([list(row) for row in unique_vertices])
            self.assertEqual(set(map(tuple, bounded_vor_vertices)), set(map(tuple, vertices)))
    
    def test_add_points_error_handling(self):
        dim = 2
        nsamples = 2 ** dim
        bounds = [[-5, 5] for d in np.arange(dim)]
        X_init, _, _, _, bounds = TestVoronoiTessellation.voronoi_initialization(dim, nsamples, bounds)

        vor = VoronoiTessellation(X_init, bounds)
        vor.build()
        initial_points = vor._all_points.copy()
        
        invalid_point = np.array([np.nan, np.nan])
        vor.add_points(invalid_point)
        same_elements, same_length = TestVoronoiTessellation.check_arrays_contain_same_elements(initial_points, vor._all_points)
        self.assertTrue(same_elements)
        self.assertTrue(same_length)
        
        invalid_point = np.array([np.inf, np.inf])
        vor.add_points(invalid_point)
        same_elements, same_length = TestVoronoiTessellation.check_arrays_contain_same_elements(initial_points, vor._all_points)
        self.assertTrue(same_elements)
        self.assertTrue(same_length)
        
        invalid_point = np.array([1.0, np.nan])
        vor.add_points(invalid_point)
        same_elements, same_length = TestVoronoiTessellation.check_arrays_contain_same_elements(initial_points, vor._all_points)
        self.assertTrue(same_elements)
        self.assertTrue(same_length)
        
        invalid_point = np.array([np.inf, 1.0])
        vor.add_points(invalid_point)
        same_elements, same_length = TestVoronoiTessellation.check_arrays_contain_same_elements(initial_points, vor._all_points)
        self.assertTrue(same_elements)
        self.assertTrue(same_length)
        
        repeat_point = X_init[0]
        vor.add_points(repeat_point)
        same_elements, same_length = TestVoronoiTessellation.check_arrays_contain_same_elements(initial_points, vor._all_points)
        self.assertTrue(same_elements)
        self.assertTrue(same_length)
        
        new_points = np.array([[1.01, 1.01, 1.01], [2.02, 2.02, 2.02], [np.nan, np.nan, 1.0]])
        with self.assertRaises(ValueError):
            vor.add_points(new_points)

        new_points = np.array([[1.01, 1.01], [np.nan, 1.0]])
        vor.add_points(new_points)
        same_elements, same_length = TestVoronoiTessellation.check_arrays_contain_same_elements(initial_points, vor._all_points)
        self.assertFalse(same_elements)
        self.assertFalse(same_length)
        
        new_points = {tuple(row) for row in new_points}
        with self.assertRaises(TypeError):
            vor.add_points(new_points)
        with self.assertRaises(TypeError):
            vor.remove_invalid_rows(new_points)

    def test_invalid_region_index_error_handling(self):
        dim = 2
        nsamples = 2 ** dim
        bounds = [[-5, 5] for d in np.arange(dim)]
        X_init, _, _, _, bounds = TestVoronoiTessellation.voronoi_initialization(dim, nsamples, bounds)

        vor = VoronoiTessellation(X_init, bounds)
        vor.build()
                       
        with self.assertRaises(ValueError):
            vor.raise_if_invalid_region_index(50)
         
        with self.assertRaises(ValueError):
            vor.raise_if_invalid_region_index(-1)
            
    def test_replace_unbounded_vertices_error_handling(self):
        dims = [2, 3]
        for dim in dims:
            nsamples = 2 ** dim
            bounds = [[-5, 5] for d in np.arange(dim)]
            X_init, _, _, _, bounds = TestVoronoiTessellation.voronoi_initialization(dim, nsamples, bounds)
            for fo in [True, False]:
                opts = {'finite_only':fo}
                vor = VoronoiTessellation(X_init, bounds)
                vor.build(**opts)
                with self.assertRaises(ValueError):
                    vor.replace_unbounded_vertices([-2, -2, -2, -2], 100, [(1, -2), (2, -3), (3, -2), (4,-2)])
                      
    def test_get_closest_seed(self):
        dims = [2, 3]
        for dim in dims:
            nsamples = 2 ** dim
            bounds = [[-5, 5] for d in np.arange(dim)]
            X_init, _, _, _, bounds = TestVoronoiTessellation.voronoi_initialization(dim, nsamples, bounds)
            vor = VoronoiTessellation(X_init, bounds)
            vor.build()
            
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
    
    def test_get_voronoi_region(self):
        from matplotlib.patches import Polygon
        from matplotlib.path import Path
        import random
        from scipy.spatial import Delaunay
        
        # Create polyhedron from region vertices
        # sample point from within polygon, and assert that point is in the region
        dims = [2, 3]
        for dim in dims:
            nsamples = 2 ** dim
            bounds = [[-5, 5] for d in np.arange(dim)]
            X_init, _, _, _, bounds = TestVoronoiTessellation.voronoi_initialization(dim, nsamples, bounds)
            vor = VoronoiTessellation(X_init, bounds)
            vor.build()

            # loop through all regions
            for region_idx, region in enumerate(vor.vor.regions):
            
                if -1 in region: # skip over regions with infinite vertices (all seeds have finite vertices)
                    continue
                if not region: # skip over empty regions
                    continue
                
                # Get region vertices
                vertices = vor.vor.vertices[region]

                # Create a Delaunay triangulation for the region vertices
                delaunay_region = Delaunay(vertices)
                
                # Define the bounding box for sampling
                min_coords = vertices.min(axis=0)
                max_coords = vertices.max(axis=0)
                
                # Sample points
                samples = np.random.uniform(min_coords, max_coords, (1000, dim))
                # Create mask for samples that are inside the polyhedron
                mask = delaunay_region.find_simplex(samples) >= 0
                inside_samples = samples[mask] 
        
                # check that get_voronoi_region returns given region
                voronoi_region_list = vor.get_voronoi_region(inside_samples)
                voronoi_region = np.array(voronoi_region_list).squeeze()
                self.assertTrue(np.all(region_idx == voronoi_region))
               
    def test_get_region_seed(self):
        dims = [2, 3]
        for dim in dims:
            nsamples = 2 ** dim
            bounds = [[-5, 5] for d in np.arange(dim)]
            X_init, _, _, _, bounds = TestVoronoiTessellation.voronoi_initialization(dim, nsamples, bounds)
            vor = VoronoiTessellation(X_init, bounds)
            vor.build()
            for pt_idx in np.arange(nsamples):
                region_index = vor.get_voronoi_region(vor.vor.points[pt_idx])[0][0]
                seed = vor.get_region_seed(region_index)
                region_point_idx, = np.where(vor.vor.point_region == region_index)
                region_seed = vor.points[region_point_idx]
                self.assertTrue(np.all(seed == region_seed))
    
    def test_find_furthest_vertex(self):
        dims = [2, 3]
        for dim in dims:        
            nsamples = 2 ** dim
            bounds = [[-5, 5] for d in np.arange(dim)]
            X_init, _, _, _, bounds = TestVoronoiTessellation.voronoi_initialization(dim, nsamples, bounds)
            vor = VoronoiTessellation(X_init, bounds)
            vor.build()
            
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
    _study_class = HaltonStudy
    
    @staticmethod
    def format_study_params_and_output(study):
        from matcal.core.data import convert_data_to_dictionary

        params_formatted = []
        for param in study.parameter_history:
            params_formatted.append(study.parameter_history[param])
        X = np.array(params_formatted).T  
    
        model_name = list(study.simulation_history.keys())[0]
        state0 = study.simulation_history[model_name].states['matcal_default_state']
        sim_history = study.simulation_history[model_name][state0]
        nsamples = len(X)
        data = []
        for nn in np.arange(nsamples):
            data.append(convert_data_to_dictionary(sim_history[nn]))
        y = data
        return X, y
    
    def setUp(self):
        super().setUp(__file__)

        # Sample data for testing
        self.model = PythonModel(quadratic_model_2d)
        self.model.set_name('quadratic_2d')
        theta1 = Parameter('a', -5, 5, distribution="uniform_uncertain")
        theta2 = Parameter('b', -5, 5, distribution="uniform_uncertain")
        pc = ParameterCollection("two_parameter", theta1, theta2)
        self.kfold = KFoldCrossValidation()
        study = self._study_class(pc)
        self.nsamples = 20
        test_points = np.linspace(.25, .75, 10)
        objective = SimulationResultsSynchronizer("x", test_points, "f")
        study.add_evaluation_set(self.model, objective)
        study_info = study.launch(self.nsamples)
        self.X, self.y = TestKFoldCrossValidation.format_study_params_and_output(study_info)
        
    def test_initialization(self):
        self.assertEqual(self.kfold.nsplits, 5)
        self.assertFalse(self.kfold.group_kfold)
        self.assertIsNone(self.kfold.scale)
        self.assertEqual(self.kfold.metric, 'rmse')
        self.assertIsNone(self.kfold.groups)
        self.assertEqual(self.kfold.interpolation_field, 'x')
        self.assertIsNone(self.kfold.par_names)

    def test_set_kfcv_options(self):
        from sklearn.linear_model import LinearRegression
        kfcv_options = {'nsplits':4, 'group_kfold':True,
                        'scale': 'cbrt', 'metric':'nlpd',
                        'groups': None,
                        'interpolation_field': 'x',
                        'par_names': ['a', 'b']
                        }
        self.kfold.X = self.X
        self.kfold._set_kfcv_options(**kfcv_options)
        self.assertEqual(self.kfold.nsplits, 4)
        self.assertTrue(self.kfold.group_kfold)
        self.assertEqual(self.kfold.scale, 'cbrt')
        self.assertEqual(self.kfold.metric, 'nlpd')
        self.assertIsNone(self.kfold.groups)
        self.assertEqual(self.kfold.interpolation_field, 'x')
        self.assertEqual(self.kfold.par_names, ['a', 'b'])
         
        # Test setting splits > number of samples. Should revert to length of X
        kfcv_options = {'nsplits': 10}
        self.kfold._set_kfcv_options(**kfcv_options)
        self.assertEqual(self.kfold.nsplits, 10)

    def test_set_kfcv_options_error_handling(self):
        kfcv_options = {'mmetric':'mse'}
        with self.assertRaises(AttributeError):
            self.kfold.perform_kfold_cv(self.X, self.y, **kfcv_options)
        kfcv_options = {'nsplits':50}
        kf_results = self.kfold._set_kfcv_options(**kfcv_options)
        self.assertTrue(self.kfold.nsplits == int(self.nsamples/2.0))
    
    def test_group_kfold_cv(self):
        nsplits = 4
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=nsplits, random_state=42)
        groups = kmeans.fit_predict(self.X)
        
        kfcv_options = {'group_kfold': True,
                        'groups': groups,
                        'nsplits': nsplits,
                        'interpolation_field': 'x',
                        'par_names': ['a', 'b']}  
        
        kf_results = self.kfold.perform_kfold_cv(self.X, self.y, **kfcv_options)
        self.assertTrue(len(kf_results) == nsplits)
        
    def test_perform_kfold_cv(self):
        kfcv_options = {'metric': 'rmse',
                        'interpolation_field':'x',
                        'par_names':['a', 'b']}
        kf_results = self.kfold.perform_kfold_cv(self.X, self.y, **kfcv_options)
       
        # Check that the results are in the expected format
        self.assertIsInstance(kf_results, dict)
        self.assertEqual(len(kf_results), self.kfold.nsplits)

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
        self.kfold.X = self.X
        self.kfold.y = self.y
        self.kfold.par_names = ['a', 'b']
        train_index = [0, 1, 2]
        test_index = [3, 4]
        error, test_idx_returned = self.kfold.evaluate_fold(train_index, test_index, self.X, self.y)
        self.assertEqual(test_idx_returned, test_index)
        self.assertIsInstance(error, float)          
       
        
class TestLeaveOneOutCrossValidation(MatcalUnitTest):
    _study_class = HaltonStudy
    
    def setUp(self):
        super().setUp(__file__)

        # Sample data for testing
        self.model = PythonModel(quadratic_model_2d)
        self.model.set_name('quadratic_2d')
        theta1 = Parameter('a', -5, 5, distribution="uniform_uncertain")
        theta2 = Parameter('b', -5, 5, distribution="uniform_uncertain")
        pc = ParameterCollection("two_parameter", theta1, theta2)
        self.loocv = LeaveOneOutCrossValidation()
        study = self._study_class(pc)
        self.nsamples = 10
        test_points = np.linspace(.25, .75, 10)
        objective = SimulationResultsSynchronizer("x", test_points, "f")
        study.add_evaluation_set(self.model, objective)
        study_info = study.launch(self.nsamples)
        self.X, self.y = TestKFoldCrossValidation.format_study_params_and_output(study_info)

    def test_initialization(self):
        self.assertIsNone(self.loocv.scale)
        self.assertEqual(self.loocv.metric, 'rmse')
        self.assertEqual(self.loocv.interpolation_field, 'x')
        self.assertIsNone(self.loocv.par_names)

    def test_set_loo_options(self):
        loocv_options = {'scale': 'cbrt', 'metric':'nlpd',
                        'interpolation_field': 'x',
                        'par_names': ['a', 'b']
                        }
        self.loocv.X = self.X
        self.loocv._set_loocv_options(**loocv_options)
        self.assertEqual(self.loocv.scale, 'cbrt')
        self.assertEqual(self.loocv.metric, 'nlpd')
        self.assertEqual(self.loocv.interpolation_field, 'x')
        self.assertEqual(self.loocv.par_names, ['a', 'b'])

    def test_set_loocv_options_error_handling(self):
        loocv_options = {'mmetric':'mse'}
        with self.assertRaises(AttributeError):
            self.loocv._set_loocv_options(**loocv_options)
         
    def test_perform_loocv(self):
        indices = range(self.nsamples)
        loocv_options = {'metric': 'rmse',
                        'interpolation_field':'x',
                        'par_names':['a', 'b']}
        loo_results = self.loocv.perform_loocv(self.X, self.y, indices, **loocv_options)
            
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


class TestVoronoiAdaptiveSurrogateStudy(MatcalUnitTest):
    
    _study_class = VoronoiAdaptiveSurrogateStudy
    _study_test_class = HaltonStudy
    
    @staticmethod
    def setup_parameter_collection(dim):
        theta1 = Parameter('a', -5, 5, distribution="uniform_uncertain")
        theta2 = Parameter('b', -5, 5, distribution="uniform_uncertain")
        if dim == 2:
            return ParameterCollection("two_parameter", theta1, theta2)
        elif dim == 3:
            theta3 = Parameter('c', -5, 5, distribution="uniform_uncertain")
            return ParameterCollection("three_parameter", theta1, theta2, theta3)

    @staticmethod
    def setup_model(dim):
        if dim == 2:
            physical_model = PythonModel(quadratic_model_2d)
            model_name = 'quadratic_2d'
        elif dim == 3:            
            physical_model = PythonModel(quadratic_model_3d)
            model_name = 'quadratic_3d'
        physical_model.set_name(model_name)
        parameter_collection =  TestVoronoiAdaptiveSurrogateStudy.setup_parameter_collection(dim)

        return physical_model, parameter_collection

    def setUp(self):
        super().setUp(__file__)
    
    def setup_test_study(self, dim):
        physical_model, parameter_collection = TestVoronoiAdaptiveSurrogateStudy.setup_model(dim)
        hal_test_study = self._study_test_class(parameter_collection)
        ntest_samples = 20

        test_points = np.linspace(.25, .75, 10)
        objective = SimulationResultsSynchronizer("x", test_points, "f")
        hal_test_study.add_evaluation_set(physical_model, objective)
        test_information = hal_test_study.launch(ntest_samples)
        return test_information
        
        
    def setup_study(self, dim):
        physical_model, parameter_collection = TestVoronoiAdaptiveSurrogateStudy.setup_model(dim)
        vor_study = self._study_class(parameter_collection)

        test_points = np.linspace(.25, .75, 10)
        objective = SimulationResultsSynchronizer("x", test_points, "f")
        vor_study.add_evaluation_set(physical_model, objective)
        return vor_study

    def test_initialization(self):
        dims = [2, 3]
        for dim in dims:
            vor_study = self.setup_study(dim)
            test_information = self.setup_test_study(dim)
            voronoi_sampling_options = {'voronoi_type':'full',
                                        'finite_only':False,
                                        'iterative_updates':True,
                                        'nmaxbatches':1,
                                        'ninitsamples':20,
                                        'seed':42}
            cross_validation_options = {'nsplits':0,
                                        'nmax_folds':3,
                                        'nmax_loo':'all',
                                        'cv_metric':'nlpd'}
            surrogate_options = {'interpolation_field':'x',
                                 'test_eval_info': test_information}
            options = {'voronoi_sampling_options': voronoi_sampling_options,
                       'surrogate_options': surrogate_options,
                       'cross_validation_options': cross_validation_options}
             
            vor_study.launch(**options)
            # build surrogate that given a, b, c gives you f
            self.assertFalse(vor_study.finite_only)
            self.assertTrue(vor_study.iterative_updates)
            self.assertEqual(vor_study.voronoi_type, 'full')
            
            if dim == 2:
                expected_boundary_points = np.array([[-5, -5],[5, -5],[-5, 5],[5, 5]])
            elif dim == 3:
                expected_boundary_points = np.array([[-5, -5, -5],
                                                     [-5, -5, 5],
                                                     [5, -5, -5],
                                                     [5, -5, 5],
                                                     [-5, 5, -5],
                                                     [-5, 5, 5],
                                                     [5, 5, -5],
                                                     [5, 5, 5]])
            self.assertTrue(np.all(vor_study._boundary_points == expected_boundary_points))
            
            from matcal.core.parameter_studies import _get_surrogate_metric
            surr_score = vor_study._current_surrogate_score['score']
            rmse = vor_study._current_surrogate_score['rmse']
            nlpd = vor_study._current_surrogate_score['nlpd']
            self.assertGreater(len(surr_score), 1)
            self.assertTrue(np.all([i > 0 for i in surr_score]))
            self.assertTrue(np.all([i > 0 for i in rmse]))
   
    def test_error_handling(self):
        test_information = self.setup_test_study(2)
        with self.assertRaises(AttributeError):
            # _voronoi_type not an attribute
            vor_study = self.setup_study(2)
            voronoi_sampling_options = {'_voronoi_type':'full',
                                        'ninitsamples':10,
                                        'seed':42}
            surrogate_options = {'interpolation_field':'x',
                                 'test_eval_info': test_information}
            options = {'voronoi_sampling_options': voronoi_sampling_options,
                       'surrogate_options': surrogate_options}
            vor_study.launch(**options)
        with self.assertRaises(ValueError):
            # random selection and thin cannot both be defined
            vor_study = self.setup_study(2)
            voronoi_sampling_options = {'thin': 10,
                                        'random_selection': 10,
                                        'ninitsamples':10,
                                        'seed':42}
            surrogate_options = {'interpolation_field':'x',
                                 'test_eval_info': test_information}
            options = {'voronoi_sampling_options': voronoi_sampling_options,
                       'surrogate_options': surrogate_options}
            vor_study.launch(**options)
        with self.assertRaises(ValueError):
            # mse not implemented
            vor_study = self.setup_study(2)
            voronoi_sampling_options = {'ninitsamples':10,
                                        'seed':42}
            cross_validation_options = {'cv_metric': 'mse'}
            surrogate_options = {'interpolation_field':'x',
                                 'test_eval_info': test_information}
            options = {'voronoi_sampling_options': voronoi_sampling_options,
                       'surrogate_options': surrogate_options,
                       'cross_validation_options': cross_validation_options}
            vor_study.launch(**options)
        with self.assertRaises(ValueError):
            # user cannot define training fraction
            vor_study = self.setup_study(2)
            surrogate_options = {'_training_fraction': 0.5,
                                 'interpolation_field':'x',
                                 'test_eval_info': test_information}
            options = {'surrogate_options': surrogate_options}             
            vor_study.launch(**options)
        with self.assertRaises(AttributeError):
            # training fraction not an attribute
            vor_study = self.setup_study(2)
            surrogate_options = {'training_fraction': 0.5,
                                 'test_eval_info': test_information}
            options = {'surrogate_options': surrogate_options}             
            vor_study.launch(**options)
        with self.assertRaises(AttributeError):
            # test info no provided
            vor_study = self.setup_study(2)
            vor_study.launch()
        with self.assertRaises(vor_study.StudyInputError):
            # user cannot add_parameter_evaluation
            vor_study.add_parameter_evaluation(a=5)
            
    def test_convergence(self):
        vor_study = self.setup_study(2)
        test_information = self.setup_test_study(2)
        voronoi_sampling_options = {'eps':10.0,
                                    'nmaxbatches': 10,
                                    'ninitsamples':20,
                                    'seed':42}
        cross_validation_options = {'nsplits':0}
        surrogate_options = {'interpolation_field':'x',
                             'test_eval_info': test_information}
        options = {'voronoi_sampling_options': voronoi_sampling_options,
                    'surrogate_options': surrogate_options,
                    'cross_validation_options': cross_validation_options}
        vor_study.launch(**options)
        converged = np.abs(vor_study._current_surrogate_score[vor_study.convergence_metric][-1] - \
                vor_study._current_surrogate_score[vor_study.convergence_metric][-2]) <= vor_study.eps
        self.assertTrue(converged)
        
    def test_nmax_loo_all(self):
        vor_study = self.setup_study(2)
        test_information = self.setup_test_study(2)
        voronoi_sampling_options = {'ninitsamples':5,
                                    'nmaxbatches':1,
                                    'seed':42}
        cross_validation_options = {'nsplits':2,
                                    'nmax_loo':'all'}
        surrogate_options = {'interpolation_field':'x',
                             'test_eval_info': test_information}
        options = {'voronoi_sampling_options': voronoi_sampling_options,
                    'surrogate_options': surrogate_options,
                    'cross_validation_options': cross_validation_options}
        vor_study.launch(**options)
        self.assertTrue(vor_study._nbatch_samples[-1] > vor_study.ninitsamples)
            
    def test_thin(self):
        vor_study = self.setup_study(2)
        test_information = self.setup_test_study(2)
        voronoi_sampling_options = {'ninitsamples':6,
                                    'nmaxbatches':1,
                                    'thin':2,
                                    'seed':42}
        cross_validation_options = {'nsplits':0}
        surrogate_options = {'interpolation_field':'x',
                             'test_eval_info': test_information}
        options = {'voronoi_sampling_options': voronoi_sampling_options,
                    'surrogate_options': surrogate_options,
                    'cross_validation_options': cross_validation_options}
        vor_study.launch(**options)
        self.assertTrue(vor_study._nbatch_samples[-1] == vor_study.ninitsamples*1.5)

    def test_random_selection(self):
        vor_study = self.setup_study(2)
        test_information = self.setup_test_study(2)
        voronoi_sampling_options = {'ninitsamples':6,
                                    'nmaxbatches':1,
                                    'random_selection':3,
                                    'seed':100}
        cross_validation_options = {'nsplits':0}
        surrogate_options = {'interpolation_field':'x',
                             'test_eval_info': test_information}
        options = {'voronoi_sampling_options': voronoi_sampling_options,
                    'surrogate_options': surrogate_options,
                    'cross_validation_options': cross_validation_options
                    }
        vor_study.launch(**options)
        self.assertTrue(vor_study._nbatch_samples[-1] == vor_study.ninitsamples*1.5)

    def test_local_tess(self):
        vor_study = self.setup_study(2)
        test_information = self.setup_test_study(2)
        voronoi_sampling_options = {'ninitsamples':20,
                                    'nmaxbatches':1,
                                    'random_selection':3,
                                    'voronoi_type':'local',
                                    'seed':42}
        cross_validation_options = {'nsplits':0}
        surrogate_options = {'interpolation_field':'x',
                             'test_eval_info': test_information}
        options = {'voronoi_sampling_options': voronoi_sampling_options,
                    'surrogate_options': surrogate_options,
                    'cross_validation_options': cross_validation_options}
        vor_study.launch(**options)
        self.assertIsNotNone(vor_study._tree)
        
    def test_group_kfold(self):
        vor_study = self.setup_study(2)
        test_information = self.setup_test_study(2)
        voronoi_sampling_options = {'ninitsamples':20,
                                    'nmaxbatches':1,
                                    'random_selection':10,
                                    'seed':42}
        cross_validation_options = {'nsplits':2,
                                    'group_kfold':True}
        surrogate_options = {'interpolation_field':'x',
                             'test_eval_info': test_information}
        options = {'voronoi_sampling_options': voronoi_sampling_options,
                    'surrogate_options': surrogate_options,
                    'cross_validation_options': cross_validation_options
                  }
        vor_study.launch(**options)
        self.assertTrue(vor_study._nbatch_samples[-1] > vor_study.ninitsamples)

    def test_perform_cv_and_find_max_errors(self):
        dims = [2, 3]
        for dim in dims:
            vor_study = self.setup_study(dim)
            test_information = self.setup_test_study(dim)
            
            nsplits = 2
            nmax_loo = 3
            voronoi_sampling_options = {'nmaxbatches':1,
                                        'seed':42}
            cross_validation_options = {'nsplits': nsplits,
                                        'nmax_folds':1,
                                        'nmax_loo': nmax_loo,
                                        'cv_metric': 'nlpd'}
            surrogate_options = {'interpolation_field':'x',
                                 'test_eval_info': test_information}
            options = {'voronoi_sampling_options': voronoi_sampling_options,
                       'surrogate_options': surrogate_options,
                       'cross_validation_options': cross_validation_options}
             
            vor_study.launch(**options)
            max_fold_indices = vor_study._max_fold_error_indices
            kf_results = vor_study._kf

            # Check that the results are in the expected format
            self.assertIsInstance(kf_results, dict)
            self.assertEqual(len(kf_results), nsplits)

            # Check that each result is a tuple (kfold error, indices of test samples)
            test_indices = []
            for result in kf_results.values():
                self.assertIsInstance(result, tuple)
                self.assertIsInstance(result[0], float)  # error
                self.assertIsInstance(result[1], np.ndarray)  # test indices
                test_indices.append(result[1])
            test_indices = np.vstack(test_indices)
            
            # check that each test sample is used once and only once
            self.assertTrue(np.all(np.isin(np.arange(vor_study.ninitsamples), test_indices)))
            self.assertEqual(len(np.unique(test_indices)), vor_study.ninitsamples)

            max_key = max(kf_results, key=kf_results.get)
            self.assertTrue(np.all(max_fold_indices == kf_results[max_key][1]))
    
            loo_errors = vor_study._loo_errors
            self.assertIsInstance(loo_errors, dict)
            self.assertEqual(len(loo_errors), len(max_fold_indices))
            for val in loo_errors.values():
                self.assertIsInstance(val, tuple)
                self.assertIsInstance(val[0], float) # error
                self.assertIsInstance(val[1], np.int64) # indice
                
            indices = [val[1] for val in loo_errors.values()]
            # check that each indice in max_fold_indices appear once and only once
            self.assertTrue(np.all(np.isin(max_fold_indices, indices)))

            loo_errors_list = [value for key, value_tuple in loo_errors.items() for value in value_tuple]
            loo_errors_array = np.asarray(loo_errors_list).reshape(-1, 2)
            sorted_array = np.asarray(sorted(loo_errors_array, key=lambda x: x[0]))

            # verify that the training samples with the greatest LOO error are returned
            worst_sample_locations = vor_study._find_loo_max_errors()
            max_error_indices = sorted_array[-nmax_loo:, :][:, 1][::-1] # get indices of largest errors and reverse (greatest to smallest)
            max_error_indices = [int(x) for x in max_error_indices] # convert entries to int
            self.assertTrue(np.all(worst_sample_locations == vor_study.X[max_error_indices]))

    def test_adaptive_voronoi_surrogate_generation(self):
        dims = [2, 3]
        for dim in dims:
            vor_study = self.setup_study(dim)
            test_information = self.setup_test_study(dim)
            
            voronoi_sampling_options = {'voronoi_type':'full',
                                        'finite_only':False,
                                        'iterative_updates':True,
                                        'nmaxbatches':3,
                                        'ninitsamples':10,
                                        'convergence_metric':'score',
                                        'seed':42}
            cross_validation_options = {'nsplits':2,
                                        'nmax_folds':1,
                                        'nmax_loo':5,
                                        'cv_metric': 'nlpd'
                                        }
            surrogate_options = {'interpolation_field':'x',
                                 'test_eval_info': test_information}
            options = {'voronoi_sampling_options': voronoi_sampling_options,
                       'surrogate_options': surrogate_options,
                       'cross_validation_options': cross_validation_options}
             
            vor_study.launch(**options)
            # verify that errors are decreasing
            metrics = [vor_study._current_surrogate_score['score'],
                      vor_study._current_surrogate_score['nlpd'],
                      vor_study._current_surrogate_score['rmse']]
            for idx, metric in enumerate(metrics):
                # error metric may not decrease every iteration. Looking for overall trend.
                metric_decreasing = metric[0] > metric[-1]
                if idx == 0:
                    self.assertFalse(metric_decreasing)
                else:
                    self.assertTrue(metric_decreasing)

            # verify that the number of samples is increasing each iteration
            nsamples = vor_study._nbatch_samples
            nsamples_increasing = all(x < y for x, y in zip(nsamples, nsamples[1:]))
            self.assertTrue(nsamples_increasing)
            
            self.assertEqual(vor_study.X.shape[0], vor_study._nbatch_samples[-1])
            
            # verify that all samples are within bounds
            samples = vor_study.X
            lb = vor_study._l_bounds
            ub = vor_study._u_bounds
            outside_samples = (samples < lb).any(axis=1) | (samples > ub).any(axis=1)
            self.assertFalse(np.any(outside_samples))
    