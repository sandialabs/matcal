from collections import OrderedDict
from copy import deepcopy
import numpy as np

from matcal.core.data import (convert_dictionary_to_data, DataCollection, 
                              ReturnPassedDataConditioner)
from matcal.core.models import PythonModel
from matcal.core.objective import (CurveBasedInterpolatedObjective, Objective, 
                                   L2NormMetricFunction, SumSquaresMetricFunction, 
                                   L1NormMetricFunction, SimulationResultsSynchronizer,
                                   DirectCurveBasedInterpolatedObjective)
                                   
from matcal.core.parameters import Parameter, ParameterCollection
from matcal.core.parameter_studies import (ClassicLaplaceStudy, FiniteDifference, 
                                           LaplaceStudy,
                                           ParameterStudy,
                                           HaltonStudy, 
                                           sample_multivariate_normal,
                                           estimate_parameter_covariance, 
                                           _get_residual_covariance, 
                                           _combine_array_list_into_zero_padded_single_array, 
                                           package_parameter_specific_results, 
                                           fit_posterior, )
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

        def model(a):
            import numpy as np
            disp = np.linspace(0,1, 10)
            load = a*disp
            return {"displacement":disp, "load":load}
        py_mod = PythonModel(model)
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

    def test_package_parameter_specific_results(self):
        param_collect = {'a':None, 'b':None}
        sens_info = {"param_dep":[0,1], "param_independent":0}
        packaged_data = package_parameter_specific_results(param_collect, sens_info)
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
            estimate_parameter_covariance(resids, resids, None)
        
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
        cov_est = estimate_parameter_covariance(resids, sens, 0)
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
            cov_est = estimate_parameter_covariance(resids, sens, 0)
        

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
        cov_est = estimate_parameter_covariance(resids, sens, noise_guess)
        print("Est covar", cov_est)
        start = np.copy(cov_est)
        print("Initial covar:", start)
        fitted_posterior = fit_posterior(resids, sens, start, noise_guess, method=None)
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
