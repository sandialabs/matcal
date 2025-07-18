from abc import ABC, abstractmethod
import numpy as np

from matcal.core.calibration_studies import ScipyMinimizeStudy
from matcal.core.data import (DataCollection, convert_dictionary_to_data, 
                              ReturnPassedDataConditioner)
from matcal.core.models import PythonModel
from matcal.core.objective import (CurveBasedInterpolatedObjective, 
                                   SimulationResultsSynchronizer, 
                                   Objective)
from matcal.core.parameters import Parameter, ParameterCollection
from matcal.core.parameter_studies import (LaplaceStudy, ParameterStudy,
                                           HaltonStudy, 
                                           ClassicLaplaceStudy, 
                                           sample_multivariate_normal)
from matcal.core.state import State
from matcal.core.tests.unit.test_study_base import (StudyBaseUnitTests, 
                                               do_nothing_load_disp_function, 
                                               model_func)


def voce_model(a, b, *args, **kwargs):
    n_x_points = 100
    xs = np.linspace(0,1.0, n_x_points)
    c = 5.0 # fixed
    ys = a*xs + b*(1.0-np.exp(-c*xs))
    return {'x':xs, 'y':ys}


def voce_model_last_point(a, b, *args, **kwargs):
    n_x_points = 100
    xs = np.linspace(0,1.0, n_x_points)
    c = 5.0 # fixed
    ys = a*xs + b*(1.0-np.exp(-c*xs))
    return {'x':[xs[-1]], 'y':[ys[-1]]}


def quadratic_function(a,b,c):
    xmax = 1.0
    npts = 100
    import numpy as np
    def f(x):
      z = a+b*x+c*x*x
      return z
    xs = np.linspace(0, xmax, npts)
    ys = f(xs)
    return {'x': xs, "y": ys}


def oneD_model(**param_dict):
    x = param_dict['theta']
    y = x ** 2
    return {"x": x, "y": y}


def twoD_multimodal(**param_dict):
    x = param_dict['x']
    y = param_dict['y']
    z = (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2
    return {'time': [0, 1], 'z': [z, z]}


class CalibrationStudyBaseUnitTests(object):
    class CalibrationCommonTests(StudyBaseUnitTests.CommonSetup, ABC):
        @abstractmethod
        def _assert_appropriate_error(self, subdir):
            """"""""
        
        @abstractmethod
        def _assert_output_created(self, subdir):
            """"""

        def setUp(self, filename):
            super().setUp(filename)
            self.path = self.get_current_files_path(__file__)

        def test_launch_with_objects_return_starting_value(self):
            study = self._study_class()
            study.add_evaluation_set(self.mock_model, self.objective, self.gold_results)
            study.set_core_limit(6)
            self._set_study_specific_options(study)
            #needed for study tests where update to parameters needed
            study.set_parameters(self.param) 

            results = study.launch()
            self.verify_results(results, self.param)

        def test_launch_with_objects_return_starting_value_in_working_dir(self):
            study = self._setup_study()
            study.add_evaluation_set(self.mock_model, self.objective, self.gold_results)
            study.set_core_limit(6)
            self._set_study_specific_options(study)
            #needed for study tests where update to parameters needed
            study.set_parameters(self.param) 
            study.set_working_directory("my_test_dir")
            results = study.launch()
            self.verify_results(results, self.param, subdir="my_test_dir")

        def test_launch_with_collections_return_starting_value(self):
            study = self._setup_study()
            study.add_evaluation_set(self.mock_model, 
                                     self.objective_collection, 
                                     self.data_collection)
            study.set_core_limit(6)
            self._set_study_specific_options(study)
            # needed for study tests where update to parameters needed
            study.set_parameters(self.parameter_collection)  
            import os
            results = study.launch()
            self.verify_results(results, self.param)

        def test_launch_with_multiple_eval_sets_per_model(self):
            study = self._setup_study()
            study.add_evaluation_set(self.mock_model, 
                                     self.objective_collection, 
                                     self.data_collection)
            study.add_evaluation_set(self.mock_model, 
                                     self.objective_collection2, 
                                     self.data_collection)
            study.add_evaluation_set(self.mock_model, 
                                     self.objective_collection3, 
                                     self.data_collection2)

            self.assertEqual(study._evaluation_sets[self.mock_model].number_of_objectives, 3)
            self.assertEqual(study._evaluation_sets[self.mock_model].residual_vector_length, 15)

            study.set_core_limit(6)
            self._set_study_specific_options(study)
            # needed for study tests where update to parameters needed
            study.set_parameters(self.parameter_collection)  
            results = study.launch()
            self.verify_results(results, self.param)

        def verify_results(self, results, param, subdir=""):
            goal_param_val = param.get_current_value() 
            self.assertAlmostEqual(results.outcome["best:a"], goal_param_val,
                                    delta=1e-5*goal_param_val)
            self._assert_output_created(subdir)
            self._assert_appropriate_error(subdir)
            
        def _setup_study(self):
            study = self._study_class(self.parameter_collection)
            study.set_results_storage_options(False, False, False, False, False)
            return study


class TestScipyMinimizeStudy(CalibrationStudyBaseUnitTests.CalibrationCommonTests):
    
    _study_class = ScipyMinimizeStudy    
        
    def _assert_appropriate_error(self, subdir):
        """"""""
    
    def _assert_output_created(self, subdir):
        """"""

    def _set_study_specific_options(self, study):
        pass
    
    def setUp(self):
        super().setUp(__file__)

class ParameterStudyTests(StudyBaseUnitTests.CommonSetup):

    _study_class = ParameterStudy

    def setUp(self):
        super().setUp(__file__)

    def test_launch_with_objects_return_starting_value(self):
        study = self._study_class(self.param)
        self.mock_model.set_name("test_mock")
        self.objective.set_name("test_objective")
        data = convert_dictionary_to_data({"displacement":np.linspace(0, 4, 6), 
                                           "load":np.linspace(1, 9, 6)})
        study.add_evaluation_set(self.mock_model, self.objective, 
                                 data)
        study.set_core_limit(6)
        self._set_study_specific_options(study)
        study.add_parameter_evaluation(a=1)
        study.add_parameter_evaluation(a=2)
        study.add_parameter_evaluation(a=3)
        study.set_results_storage_options(True, True, True, True, True)
        
        results = study.launch()
        model_results = convert_dictionary_to_data(do_nothing_load_disp_function(1))

        gold_flattened_residuals = (np.interp(data["displacement"], model_results["displacement"], 
                                              model_results["load"])/np.max(data["load"])-
                                    data["load"]/np.max(data["load"]))
        first_eval = results.evaluation_sets[0]
        calc_res_all = results.objective_history[first_eval].weighted_conditioned_residuals

        for i in range(3):    
            calc_res = calc_res_all[i][data.state][0] 
            self.assert_close_arrays(gold_flattened_residuals,calc_res)
            self.assertAlmostEqual(results.parameter_history['a'][i], i+1)

    def test_launch_with_objects_return_starting_value_no_data(self):
        coeff = Parameter("coeff", 0,4)
        power = Parameter("power", 0,4)
        offset = Parameter("offset", 0,4)
        
        study = self._study_class(coeff, power, offset)
        model = PythonModel(model_func)
        obj = SimulationResultsSynchronizer("x", np.linspace(0, 10, 100), "y")
        obj.set_name("test_objective")
        study.add_evaluation_set(model, obj)
        study.set_core_limit(6)
        self._set_study_specific_options(study)
        study.add_parameter_evaluation(coeff=1, power=1, offset=0)
        study.add_parameter_evaluation(coeff=2, power=1, offset=0)
        study.add_parameter_evaluation(coeff=3, power=1, offset=0)
        study.set_results_storage_options(True, True, True, True, True)
        results = study.launch()
              
        gold_flattened_residuals = []
        gold_flattened_residuals.append(model_func(1, 1, 0)["y"])
        gold_flattened_residuals.append(model_func(2, 1, 0)["y"])
        gold_flattened_residuals.append(model_func(3, 1, 0)["y"])
        first_eval = results.evaluation_sets[0]
        all_calc_res =results.objective_history[first_eval].weighted_conditioned_residuals

        for i in range(3):
            calc_res = all_calc_res[i]['matcal_default_state']
            self.assert_close_arrays(gold_flattened_residuals[i],calc_res)
            self.assertAlmostEqual(results.parameter_history["coeff"][i], i+1)
                                         

class ClassicLaplaceStudyTests(StudyBaseUnitTests.CommonSetup):

    _study_class = ClassicLaplaceStudy

    def setUp(self):
        super().setUp(__file__)

    def test_evaluate_finite_difference_stencil(self):
        b = Parameter("b",0,2, 1.1)
        c = Parameter("c",0,2, 1.3)
        parameters = ParameterCollection("test params", b, c)
        laplace_study = self._study_class(parameters)

        state_1 = State("one", a = 2)
        state_2 = State("two", a = -1)

        data_collection = DataCollection("test")
        npts = 100
        noise_stddev = 0.01
        np.random.seed(1090312)
        noise = np.random.normal(0, noise_stddev, npts)
        data_state_1 =  convert_dictionary_to_data(
            quadratic_function(b=b.get_current_value(), c=c.get_current_value(), 
                               **state_1.params))
        data_state_1.set_state(state_1)
        data_state_1["y"] = data_state_1["y"]+noise
        data_collection.add(data_state_1)

        data_state_2 =  convert_dictionary_to_data(
            quadratic_function(b=b.get_current_value(), c=c.get_current_value(),
                                **state_2.params))
        data_state_2.set_state(state_2)
        data_state_2["y"] = data_state_2["y"]+noise 
        data_collection.add(data_state_2)
            
        model = PythonModel(quadratic_function)
        model.set_name("test_model")

        objective = CurveBasedInterpolatedObjective("x", "y")
        obj_name = "test_obj"
        objective.set_name(obj_name)

        cal_study = ScipyMinimizeStudy(parameters)
        cal_study.add_evaluation_set(model, objective ,
                                         data_collection)
        cal_res = cal_study.launch()
        centers = {'b': cal_res.best.b, 'c': cal_res.best.c}
        
        laplace_study.add_evaluation_set(model, objective ,
                                         data_collection, 
                                         data_conditioner_class=ReturnPassedDataConditioner)

        laplace_study.set_parameter_center(**centers)
        laplace_study.set_core_limit(6)
        laplace_study.set_step_size(1e-5)
        results = laplace_study.launch()
        laplace_gradient_b = results.objective_gradient.b
        laplace_gradient_c = results.objective_gradient.c
        
        laplace_hessian  = results.hessian
        laplace_covariance = results.estimated_parameter_covariance

        xs = np.linspace(0,1,npts)
        # there are two states, so we need to stack the xs twice
        xs = np.column_stack((xs,xs))
        #second derivatives of sum of squared obj (sum((data-(f))^2))
        # are independent of b/c
        hes_goal_b_b = 2*np.sum(xs*xs)
        hes_goal_b_c = 2*np.sum(xs*xs*xs)
        hes_goal_c_c = 2*np.sum(xs*xs*xs*xs)
        hessian_goal = np.array([[hes_goal_b_b, hes_goal_b_c], 
                                 [hes_goal_b_c, hes_goal_c_c]])

        covariance_goal = 2*noise_stddev*noise_stddev*np.linalg.inv(hessian_goal)
        self.assertAlmostEqual(laplace_gradient_b, 
                                0.0, delta=1e-2)        
        self.assertAlmostEqual(laplace_gradient_c, 
                                 0.0, delta=1e-2)
        #need to scale by 200 (we scale by the length and number of states)
        self.assert_close_arrays(laplace_hessian*200, hessian_goal, show_on_fail=True)        
        self.assert_close_arrays(laplace_covariance, covariance_goal, 
                                 rtol = 0.05, show_on_fail=True)

    def test_external_noise_uq(self):
        b = Parameter("b",0,2, 1.1)
        c = Parameter("c",0,2, 1.3)
        parameters = ParameterCollection("test params", b, c)
        laplace_study = self._study_class(parameters)

        state_1 = State("one", a = 2)
        state_2 = State("two", a = -1)

        data_collection = DataCollection("test")
        npts = 100
        noise_stddev = 0.01
        np.random.seed(1090312)
        noise = np.random.normal(0, noise_stddev, npts)
        data_state_1 =  convert_dictionary_to_data(
            quadratic_function(b=b.get_current_value(), c=c.get_current_value(), 
                               **state_1.params))
        data_state_1.set_state(state_1)
        data_state_1["y"] = data_state_1["y"]+noise
        data_collection.add(data_state_1)

        data_state_2 =  convert_dictionary_to_data(
            quadratic_function(b=b.get_current_value(), c=c.get_current_value(),
                                **state_2.params))
        data_state_2.set_state(state_2)
        data_state_2["y"] = data_state_2["y"]+noise 
        data_collection.add(data_state_2)
            
        model = PythonModel(quadratic_function)
        model.set_name("test_model")

        objective = CurveBasedInterpolatedObjective("x", "y")
        obj_name = "test_obj"
        objective.set_name(obj_name)

        cal_study = ScipyMinimizeStudy(parameters)
        cal_study.add_evaluation_set(model, objective ,
                                         data_collection)
        cal_res = cal_study.launch()
        centers = {'b': cal_res.best.b, 'c': cal_res.best.c}
        
        laplace_study.add_evaluation_set(model, objective ,
                                         data_collection, 
                                         data_conditioner_class=ReturnPassedDataConditioner)

        laplace_study.set_parameter_center(**centers)
        laplace_study.set_core_limit(6)
        laplace_study.set_step_size(1e-5)
        results = laplace_study.launch()

        xs = np.linspace(0,1,npts)
        # there are two states, so we need to stack the xs twice
        xs = np.column_stack((xs,xs))
        #second derivatives of sum of squared obj (sum((data-(f))^2))
        # are independent of b/c
        hes_goal_b_b = 2*np.sum(xs*xs)
        hes_goal_b_c = 2*np.sum(xs*xs*xs)
        hes_goal_c_c = 2*np.sum(xs*xs*xs*xs)
        hessian_goal = np.array([[hes_goal_b_b, hes_goal_b_c], 
                                 [hes_goal_b_c, hes_goal_c_c]])
        covariance_goal = 2*noise_stddev*noise_stddev*np.linalg.inv(hessian_goal)

        nsamples=5000
        samples = sample_multivariate_normal(nsamples, results.mean.to_list(), 
                                           results.estimated_parameter_covariance, 
                                           param_names=["b", "c"], 
                                           seed=1090312)
        self.assertEqual(len(samples["b"]), nsamples)
        self.assertEqual(len(samples["c"]), nsamples)
        param_std_dev_goal = np.array([np.sqrt(covariance_goal[0,0]), 
                                       np.sqrt(covariance_goal[1,1])])
        param_std_dev = np.array([np.std(samples["b"]), np.std(samples["c"])])
        self.assert_close_arrays(param_std_dev, param_std_dev_goal, 5e-4, 
                                 show_on_fail=True)
        

class LaplaceStudyTests(StudyBaseUnitTests.CommonSetup):

    _study_class = LaplaceStudy

    def setUp(self):
        super().setUp(__file__)

    def test_uq(self):
        n_experiments = 750
        mu_theta = np.array([2.0, 1.0])
        var_theta = np.array([[0.4,0.05],[0.05,0.1]])

        measurement_noise = 0.0
        mod = PythonModel(voce_model)

        def generate_data(noise=None):
            rng = np.random.default_rng(seed=123456)
            thetas = rng.multivariate_normal(mu_theta, var_theta, size=n_experiments)
            dc = DataCollection("generated data")
            for theta in thetas:
                mod_res = voce_model(theta[0], theta[1])
                if noise is not None:
                    mod_res['y'] += noise*np.random.normal(size=mod_res['y'].shape)
                dc.add(convert_dictionary_to_data(mod_res))
            return dc
        dc = generate_data(measurement_noise)

        a = Parameter("a",0,5)
        b = Parameter("b",0,5)
        cal = ScipyMinimizeStudy(a,b)
        obj = Objective('y')
        obj.set_as_large_data_sets_objective()
        cal.add_evaluation_set(mod, obj, dc)

        results = cal.launch()
        a_min = results.best.a
        b_min = results.best.b

        study = self._study_class(a,b)
        study.set_parameter_center(a=a_min, b=b_min)
        laplace_obj = CurveBasedInterpolatedObjective('x', 'y')
        study.add_evaluation_set(mod, laplace_obj, dc)
        study.set_noise_estimate(measurement_noise)
        results = study.launch()

        self.assert_close_arrays(var_theta, results.estimated_parameter_covariance,  
                                 atol = 0.0075)    
        
    def test_uq_two_models(self):
        n_experiments = 500
        
        mu_theta = np.array([2.0, 1.0, 0.5])
        var_theta = np.array([[0.4, 0.05, 0.2],
                              [0.05, 0.1, 0.1], 
                              [0.2, 0.1, 0.2]])

        measurement_noise = 0.0
        voce_mod = PythonModel(voce_model)
        voce_mod.set_name("voce")
        quadratic_mod = PythonModel(quadratic_function)
        quadratic_mod.set_name("quadratic")
        def generate_data(noise=None):
            rng = np.random.default_rng(seed=123456)
            thetas = rng.multivariate_normal(mu_theta, var_theta, size=n_experiments)
            dc_voce = DataCollection("voce generated data")
            dc_quadratic = DataCollection("quadratic generated data")

            for theta in thetas:
                voce_res = voce_model(theta[0], theta[1])
                quad_res = quadratic_function(theta[0], theta[1], theta[2])
                if noise is not None:
                    voce_res['y'] += noise*np.random.normal(size=voce_res['y'].shape)
                    quad_res['y'] += noise*np.random.normal(size=quad_res['y'].shape)

                dc_voce.add(convert_dictionary_to_data(voce_res))
                dc_quadratic.add(convert_dictionary_to_data(quad_res))

            return dc_voce, dc_quadratic
        dc_voce, dc_quadratic = generate_data(measurement_noise)

        a = Parameter("a",0,5)
        b = Parameter("b",0,5)
        c = Parameter("c",0,5)

        cal = ScipyMinimizeStudy(a,b,c)
        obj = Objective('y')
        obj.set_name("x-y obj")
        cal.add_evaluation_set(voce_mod, obj, dc_voce)
        cal.add_evaluation_set(quadratic_mod, obj, dc_quadratic)
        results = cal.launch()
        a_min = results.best.a
        b_min = results.best.b
        c_min = results.best.c

        study = self._study_class(a,b,c)
        study.set_parameter_center(a=a_min, b=b_min, c=c_min)
        laplace_obj = CurveBasedInterpolatedObjective('x', 'y')
        study.add_evaluation_set(voce_mod, laplace_obj, dc_voce)
        study.add_evaluation_set(quadratic_mod, laplace_obj, dc_quadratic)
        
        study.set_noise_estimate(measurement_noise)
        results = study.launch()

        self.assert_close_arrays(var_theta, results.estimated_parameter_covariance,  
                                 atol = 0.05)    

    def test_uq_with_single_value_obj(self):
        n_experiments = 250
        mu_theta = np.array([2.0, 1.0])
        var_theta = np.array([[0.4,0.05],[0.05,0.1]])

        measurement_noise = 0.0
        mod = PythonModel(voce_model)
        single_point_mod = PythonModel(voce_model_last_point)

        def generate_data(noise=None):
            rng = np.random.default_rng(seed=123456)
            thetas = rng.multivariate_normal(mu_theta, var_theta, size=n_experiments)
            dc = DataCollection("generated data")
            dc_single_point = DataCollection("single point data")
            for theta in thetas:
                mod_res = voce_model(theta[0], theta[1])
                if noise is not None:
                    mod_res['y'] += noise*np.random.normal(size=mod_res['y'].shape)
                dc.add(convert_dictionary_to_data(mod_res))
                dc_single_point.add(convert_dictionary_to_data({'y':[mod_res['y'][-1]]}))
            return dc, dc_single_point
        dc, dc_single_point = generate_data(measurement_noise)

        a = Parameter("a",0,5)
        b = Parameter("b",0,5)
        cal = ScipyMinimizeStudy(a,b)
        obj = Objective('y')
        cal.add_evaluation_set(mod, obj, dc)
        cal.add_evaluation_set(single_point_mod, obj, dc_single_point)

        results = cal.launch()
        a_min = results.best.a
        b_min = results.best.b

        study = self._study_class(a,b)
        study.set_parameter_center(a=a_min, b=b_min)
        laplace_obj = CurveBasedInterpolatedObjective('x', 'y')
        study.add_evaluation_set(mod, laplace_obj, dc)        
        study.add_evaluation_set(single_point_mod, obj, dc_single_point)

        study.set_noise_estimate(measurement_noise)
        results = study.launch()

        self.assert_close_arrays(var_theta, results.estimated_parameter_covariance,  
                                 atol = 0.05)  

class TestHaltonStudy(StudyBaseUnitTests.CommonTests):

    _study_class = HaltonStudy

    @staticmethod
    def setup_1d_parameter_collection():
        theta = Parameter("theta", -2, 2, distribution="uniform_uncertain")
        return ParameterCollection("one_parameter", theta)
        
    @staticmethod
    def setup_2d_parameter_collection():
        x = Parameter("x", -5, 5, distribution="uniform_uncertain")
        y = Parameter("y", -5, 5, distribution="uniform_uncertain")
        return ParameterCollection("two_parameter", x, y)

    @staticmethod
    def run_study(study, nsamples, model_name, par_names, skip=None):
        results = study.launch(nsamples, skip=skip)
        params = np.array([results.parameter_history[par] for par in par_names]).T.squeeze()
        state0 = results.simulation_history[model_name].states['matcal_default_state']
        sim_history = results.simulation_history[model_name][state0]
        return params, sim_history, state0

    @staticmethod
    def calculate_nd_interpolated_pred_error(params, data, test_points, test_data):
        from scipy.interpolate import LinearNDInterpolator
        
        interp = LinearNDInterpolator(params, np.array(data), fill_value=np.nan, rescale=False)
        y_pred = interp(test_points)
        diff = y_pred - test_data
        return np.linalg.norm(np.nan_to_num(diff)), y_pred
    
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

    def test_1d_halton_distribution(self):
        from scipy.stats import kstest
        
        nsamples = 100
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
        params, _, _ = TestHaltonStudy.run_study(study, nsamples, model_name, par_names)
        param_array = np.array(params)
        scaled_params = (param_array - param_array.min()) / (param_array.max() - param_array.min())
        _, p_value = kstest(scaled_params, 'uniform')
        
        # check if parameters follow uniform distribution
        # Null hypothesis that there is no difference between the distribution of params and a uniform distribution.
        self.assertGreater(p_value, 0.05)
            
    def test_2d_launch(self):
        from scipy.interpolate import interp1d
        import matplotlib.pyplot as plt
        nsamples = 200
        model_name = '2d_multimodal'
        par_names = ['x', 'y']
        
        # set up model, objective, paramter collection
        model = PythonModel(twoD_multimodal)
        model.set_name('2d_multimodal')
        parameter_collection = TestHaltonStudy.setup_2d_parameter_collection()
        
        xi = np.linspace(-5, 5, 15)
        yi = np.linspace(-5, 5, 15)
        xv, yv = np.meshgrid(xi, yi)
        
        test_points = np.array([xv.ravel(), yv.ravel()]).T 

        # push, pull current, rebase off of current
        objective = SimulationResultsSynchronizer("time", [0], "z")
        
        # run initial study
        study = self.setup_study(parameter_collection, model, objective)
        params, sim_history, state0 = TestHaltonStudy.run_study(study, nsamples, model_name, par_names)
        self.assertEqual(params.shape[0], nsamples)
        self.assertEqual(params.shape[1], 2)
        
        data = [sim_history[i]['z'][0] for i in range(nsamples)]

        test_data = []
        import time
        for val in test_points: 
            x = Parameter("x", -5, 5, distribution="uniform_uncertain", current_value=val[0])
            y = Parameter("y", -5, 5, distribution="uniform_uncertain", current_value=val[1])
            pc = ParameterCollection("predictions", x, y)
            res = model.run(state0, pc)
            test_data.append(res.results_data['z'][0])
            time.sleep(0.1)
        test_data = np.array(test_data)
         
        # interpolate and calculate prediction error of test points
        pred_error, y_pred = TestHaltonStudy.calculate_nd_interpolated_pred_error(\
            params, data, test_points, test_data)

        # continue study with additional Halton samples
        nnew_samples = 200
        study = self.setup_study(parameter_collection, model, objective)
        study.restart()
        new_params, sim_history, state0 = TestHaltonStudy.run_study(study, nsamples+nnew_samples, model_name, par_names)
        self.assertEqual(new_params.shape[0], nsamples + nnew_samples)
        self.assertEqual(new_params.shape[1], 2)

        new_data = [sim_history[i]['z'][0] for i in range(nsamples+nnew_samples)]

        # interpolate and calculate prediction error of test points
        new_pred_error, y_pred = TestHaltonStudy.calculate_nd_interpolated_pred_error(\
            new_params, new_data, test_points, test_data)
        
        # prediction error should be less with more Halton samples
        self.assertGreater(pred_error, new_pred_error)


