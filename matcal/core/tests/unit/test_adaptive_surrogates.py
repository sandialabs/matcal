import os
import types
import numpy as np
import shutil
import unittest

from matcal.core.adaptive_surrogates import (
    SparseGridAdaptiveSurrogateStudy,
    _get_parameter_bounds, SparseGridAdaptiveSurrogate,
    VoronoiAdaptiveSurrogateStudy, VoronoiTessellation, 
    LeaveOneOutCrossValidation, KFoldCrossValidation)
from matcal.core.data import convert_dictionary_to_data
from matcal.core.objective import SimulationResultsSynchronizer
from matcal.core.models import PythonModel
from matcal.core.parameters import Parameter, ParameterCollection
from matcal.core.parameter_studies import HaltonStudy
from matcal.core.qoi_extractor import UserDefinedExtractor, InterpolatingExtractor
from matcal.core.serializer_wrapper import matcal_load
from matcal.core.tests.MatcalUnitTest import MatcalUnitTest


try:
    import pyapprox  
    HAS_PYAPPROX = True
except Exception:    
    HAS_PYAPPROX = False


def change_y_to_z(working_data, reference_data, return_keys_list):
    working_qois = {}
    for key in working_data.field_names:
        if key == "y":
            working_qois["z"] = working_data[key]
        else:
            working_qois[key] = working_data[key]
    working_qois = convert_dictionary_to_data(working_qois)
    interp = InterpolatingExtractor("x")
    working_qois = interp.calculate(working_qois, reference_data, return_keys_list)
    return working_qois


def return_data(*args, **kwargs):
    return {"x":np.linspace(0,1,10), "y":np.linspace(1,3,10)}
light_model = PythonModel(return_data)
light_model.set_name("light_model")


def linear_model_2d(**parameters):
    """Linear curve: f = a + bx. x independent variable."""
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


a = Parameter("a", 0, 10)
b = Parameter("b", 0, 10)
c = Parameter("c", 0.1, 2)


class TestSparseGridAdaptiveSurrogateStudy(MatcalUnitTest):
    """
    All tests inherit from MatcalUnitTest so that they get the same
    temporary build directory handling that the original MatCal unit tests use.
    """

    def setUp(self):
        super().setUp(__file__)           
        p1 = Parameter("p1", 0.0, 1.0, 0.5)
        p2 = Parameter("p2", -1.0, 1.0, 0.0)
        self.simple_parameters =(p1, p2)
        self.study = SparseGridAdaptiveSurrogateStudy(*self.simple_parameters)

    def test_parameter_bounds(self):
        bounds = _get_parameter_bounds(self.study._parameter_collection)
        expected = np.array([[0.0, 1.0],
                             [-1.0, 1.0]])
        np.testing.assert_array_equal(bounds, expected)

    def test_set_independent_variable(self):
        self.study.set_independent_variable("time", np.linspace(0, 1, 5))
        self.assertEqual(self.study._independent_variable, "time")
        np.testing.assert_array_equal(
            self.study._independent_variable_values,
            np.linspace(0, 1, 5)
        )
        with self.assertRaises(TypeError):
            self.study.set_independent_variable(1.0, np.linspace(0,1,2))

        with self.assertRaises(ValueError):
            self.study.set_independent_variable("", "")

        with self.assertRaises(TypeError):
            self.study.set_independent_variable("1", "")
    
    def test_error_with_add_parameter_set(self):
        with self.assertRaises(RuntimeError):
            self.study.add_parameter_evaluation(p1=1, p2=2)

    def test_set_target_field(self):
        self.study.set_target_field_name("temperature")
        self.assertEqual(self.study._target_field_name, "temperature")

        with self.assertRaises(TypeError):
            self.study.set_target_field_name(1.0)

    def test_set_number_of_test_samples(self):
        self.study.set_number_of_test_samples(20)
        self.assertEqual(self.study._number_of_test_samples, 20)
        with self.assertRaises(TypeError):
            self.study.set_number_of_test_samples("a")

    def test_set_number_max_training_samples(self):
        self.study.set_max_training_samples(500)
        self.assertEqual(self.study._max_training_samples, 500)

        with self.assertRaises(TypeError):
            self.study.set_max_training_samples("a")

    def test_set_group_random_seed(self):
        #appears to be no way to really test the seed that gets into the SciPy Halton class.
        self.study.set_seed(10)
        self.study.set_test_group_random_seed(1234)
        self.assertEqual(self.study._test_group_random_seed, 1234)
        self.assertEqual(self.study._seed, 10)

        with self.assertRaises(TypeError):
            self.study.set_test_group_random_seed("")
        
    def test_default_test_samples_calculation(self):
        self.study.set_max_training_samples(200)# 200//20 = 10; 2 params*10 = 20 → default = 20
        default = self.study._set_default_number_of_test_samples()
        self.assertEqual(default, 20)
        
        self.study.set_max_training_samples(2000)# 2000//20 = 100; 2 params*10 = 20 → default = 100
        default = self.study._set_default_number_of_test_samples()
        self.assertEqual(default, 100)

    def test_make_simulation_results_synchronizer_success(self):
        self.study.set_independent_variable("x", [0.0, 1.0])
        self.study.set_target_field_name("y")
        sync = self.study._make_simulation_results_synchronizer(None)
        self.assertIsInstance(sync, SimulationResultsSynchronizer)
        self.assertEqual(sync.independent_field, "x")
        self.assertEqual(sync._independent_field_values, [0.0, 1.0])
        self.assertEqual(sync.fields_of_interest, ("y",))

    def test_make_simulation_results_synchronizer_missing(self):
        self.study.set_independent_variable("x", [0, 1])
        with self.assertRaises(RuntimeError) as ctx:
            self.study._make_simulation_results_synchronizer(None)
        self.assertIn("Target field name", str(ctx.exception))

        self.study._independent_variable = None
        self.study._independent_variable_values = None
        self.study.set_target_field_name("test")
        with self.assertRaises(RuntimeError) as ctx:
            self.study._make_simulation_results_synchronizer(None)
        self.assertIn("Independent variable name", str(ctx.exception))

    def test_add_evaluation_set_once(self):
        """First call succeeds, second call raises."""
        self.study.set_independent_variable("t", [0, 1])
        self.study.set_target_field_name("z")
        # First call – ok
        self.assertFalse(self.study._evaluation_set_added)
        self.study.add_evaluation_set(light_model)
        self.assertTrue(self.study._evaluation_set_added)

        # Second call – must raise
        with self.assertRaises(RuntimeError):
            self.study.add_evaluation_set(light_model)

    def test_add_evaluation_set_invalid_state(self):
        self.study.set_independent_variable("t", [0, 1])
        self.study.set_target_field_name("z")
        with self.assertRaises(TypeError):
            self.study.add_evaluation_set(light_model, state="not_a_state")

    def test_update_work_dir_for_test_sampling(self):
        original = self.study._working_directory
        returned = self.study._update_work_dir_for_test_sampling()
        self.assertEqual(returned, original)
        self.assertEqual(self.study._working_directory, os.path.abspath("test_samples"))

        self.study.set_working_directory("work")
        returned = self.study._update_work_dir_for_test_sampling()
        self.assertEqual(os.path.abspath("work"), returned)

        expected = os.path.abspath("work" + "_test_samples")
        self.assertEqual(os.path.abspath(self.study._working_directory), expected)

    @unittest.skipIf(not HAS_PYAPPROX,
                 "pyapprox not installed – skipping pyapprox‑dependent tests")
    def test_launch_creates_test_directory_and_restores(self):
        self.study.set_independent_variable("x", np.linspace(0.0, 1.0, 4))
        self.study.set_target_field_name("y")
        self.study.set_max_training_samples(1)
        self.study.set_number_of_test_samples(1)
        self.study.add_evaluation_set(light_model)
        self.study.launch()
        test_dir = os.path.abspath("test_samples")
        self.assertTrue(os.path.isdir(test_dir))

        self.study._reset_study_after_test_sampling_generation(None, True)
        self.study.set_working_directory("work")
        test_dir = os.path.abspath("work_test_samples")
        self.study.launch()
        self.assertTrue(os.path.isdir(test_dir))
        self.assertTrue(os.path.isdir("work"))

    @unittest.skipIf(not HAS_PYAPPROX,
                 "pyapprox not installed – skipping pyapprox‑dependent tests")
    def test_user_qoi_extractor_in_add_eval_set(self):
        self.study.set_independent_variable("x", np.linspace(0.0, 1.0, 4))
        self.study.set_target_field_name("z")
        self.study.set_max_training_samples(1)
        self.study.set_number_of_test_samples(1)
        extractor = UserDefinedExtractor(change_y_to_z, "y", "x")
        with self.assertRaises(TypeError):
            self.study.add_evaluation_set(light_model, qoi_extractor="yay")
        self.study.add_evaluation_set(light_model, qoi_extractor=extractor)
        results = self.study.launch()
        objs = results.get_objectives_for_model("light_model")
        self.assertTrue("z" in results.best_simulation_qois("light_model", objs[0], 
                                                             "matcal_default_state", 0).field_names)

    def test_format_params_and_output(self):
        """Check that the two formatting helpers produce the expected arrays."""
        class FakeResults:
            def __init__(self):
                self.parameter_history = {"p1": [0.1, 0.2, 0.4],
                                          "p2": [0.3, 0.4, 0.25]}
                self.number_of_evaluations = 3
                self.simulation_history = {}
                self.qoi_history = {}

        fake = FakeResults()

        params = self.study._format_params(fake)
        self.assertIsInstance(params, np.ndarray)
        self.assertTupleEqual(params.shape, (3, 2))

        model_name = "dummy_model"

        class DummyObjective:
            def __init__(self, name):
                self.name = name
        dummy_obj = DummyObjective("obj")

        class DummySimQoi:
            def __init__(self, state_name, target, values):
                self._data = {state_name: [{target: np.array(values)}]}

            def __getitem__(self, key):
                return self._data[key]

        state_name = "state0"
        target = "temperature"
        sim0 = DummySimQoi(state_name, target, [1.0, 2.0])
        sim1 = DummySimQoi(state_name, target, [3.0, 4.0])
        sim2 = DummySimQoi(state_name, target, [5.0, 6.0])

        class DummyQoiContainer:
            def __init__(self, sim_qois):
                self.simulation_qois = sim_qois

        fake.qoi_history[f"{model_name}:{dummy_obj.name}"] = DummyQoiContainer([sim0, sim1, sim2])

        # simulation_history must provide state_names
        fake.simulation_history = {
            model_name: types.SimpleNamespace(state_names=[state_name])
        }

        self.study.set_independent_variable("time", np.array([0.0, 1.0]))
        self.study.set_target_field_name(target)

        # Monkey‑patch the private helpers that would otherwise inspect the real
        # evaluation‑set collections.
        self.study._get_model_names = lambda: [model_name]
        self.study._results_synchronizer = dummy_obj

        # ----- _format_output -----
        output = self.study._format_output(fake)
        self.assertIsInstance(output, np.ndarray)
        self.assertTupleEqual(output.shape, (3, 2))

        expected = np.array([[1.0, 2.0],
                             [3.0, 4.0], 
                             [5.0, 6.0]])
        np.testing.assert_array_equal(output, expected)

    def test_format_batch_results_returns_expected_array(self):
        class DummyQoiInfo:
            def __init__(self, state, target, values):
                # store as: {state: [{target: np.array(values)}]}
                self.simulation_qois = {
                    state: [{target: np.array(values)}]
                }

        sample_values = [
            [10.0, 20.0, 30.0],  
            [40.0, 50.0, 60.0],  
            [70.0, 80.0, 90.0]   
        ]

        self.study.set_independent_variable("x", np.linspace(0,1,3))
        self.study.set_target_field_name("y")
        self.study.add_evaluation_set(light_model)
        state_name = "state"

        class FakeResults:
            class FakeDC:
                def __init__(self, state):
                    self.state_names = [state]

            def __init__(self,state):
                self.simulation_history = {light_model.name:self.FakeDC(state)}

        self.study._results = FakeResults(state_name)
        target = "y"
        model_name = light_model.name
        qois_list = []
        for vals in sample_values:
            qoi_obj = DummyQoiInfo(state_name, target, vals)
            # The outer dict is {model_name: {objective_name: qoi_obj}}
            qois_list.append({model_name: {self.study._results_synchronizer.name: qoi_obj}})

        batch_results = {"qois": qois_list}

        n_params = 2
        n_samples = len(sample_values)          # 3 samples
        param_sets = np.zeros((n_params, n_samples))

        formatted = self.study._format_batch_results(batch_results, param_sets)

        expected = np.array(sample_values, dtype=float)

        self.assertIsInstance(formatted, np.ndarray)
        self.assertEqual(formatted.shape, (n_samples, len(self.study._independent_variable_values)))
        np.testing.assert_array_equal(formatted, expected)

    def test_populate_parameter_evaluations_adaptive(self):
        """
        Verify that the method:
        * treats the sample matrix as native/physical parameter values,
        * creates a list ``_parameter_sets_to_evaluate`` with one dict per
            sample,
        * each dict contains the correct parameter names and values, and
        * ``_add_parameter_evaluation`` is called with the same dictionaries.
        """
        native_samples = np.array([
            [0.1, 0.2, 0.3],    # p1 values
            [0.1, 0.5, -0.1],   # p2 values
        ])

        self.study._populate_parameter_evaluations_adaptive(native_samples)
        self.assertEqual(len(self.study._parameter_sets_to_evaluate), 3)

        expected_names = self.study._parameter_collection.get_item_names()
        expected_names = list(expected_names)

        for i, param_dict in enumerate(self.study._parameter_sets_to_evaluate):
            self.assertCountEqual(param_dict.keys(), expected_names)
            for name, col_idx in zip(expected_names, range(len(expected_names))):
                expected_value = native_samples[col_idx, i]
                self.assertAlmostEqual(param_dict[name], expected_value)

    def test_update_surrogate_score(self):
        self.study._test_params = np.array([[0.0, 1.0],
                                            [0.5, 0.5],
                                            [1.0, 0.0]])
        self.study._test_responses = np.array([[1.0, 2.0],
                                            [1.5, 2.5],
                                            [0.75, 0.85]])

        class DoubleParamSurrogate:
            def __init__(self):
                self.surrogate = self

            def __call__(self, params):
                results = np.array(params) * 2.0
                return results

        self.study._surrogate = SparseGridAdaptiveSurrogate(
            "target",
            "independent",
            np.array([0.0, 1.0]),
            self.study._test_params,
            self.study._test_responses,
            ["p1", "p2"],
            np.array([[-1.0, 1.0], [-1.0, 1.0]]),
        )

        self.study._surrogate._add_iteration(DoubleParamSurrogate(), 2)

        self.assertEqual(len(self.study.surrogate.rmse_history), 1)
        self.assertEqual(len(self.study.surrogate.max_error_history), 1)

        expected_rmse = np.sqrt(
            np.mean(
                (
                    self.study._test_responses
                    - DoubleParamSurrogate()(self.study._test_params.T).T
                ) ** 2
            )
        )
        self.assertAlmostEqual(self.study.surrogate.rmse_history[0], expected_rmse)

    @unittest.skipIf(not HAS_PYAPPROX,
                 "pyapprox not installed – skipping pyapprox‑dependent tests")
    def test_study_results_is_not_none(self):
        """launch must eventually call the sparse‑grid routine."""
        self.study.set_independent_variable("x", np.linspace(0,1,3))
        self.study.set_target_field_name("y")
        self.study.add_evaluation_set(light_model)
        self.study.set_max_training_samples(30)
        self.study.set_number_of_test_samples(1)
        self.study.launch()
        self.assertIsNotNone(self.study.results)
        sur_file = self.study.surrogate_save_filename
        sur_results = matcal_load(sur_file)
        self.assertEqual(self.study.surrogate.rmse_history, 
                         sur_results.rmse_history)
        self.assertEqual(self.study.surrogate.sample_count_history, 
                         sur_results.sample_count_history)
        self.assertEqual(self.study.surrogate.max_error_history, 
                         sur_results.max_error_history) 
        self.assertEqual(len(self.study.surrogate._surrogates), 
                         len(sur_results._surrogates))
        self.assert_close_dicts_or_data(sur_results(p1=0, p2=0), 
                                        self.study.surrogate(p1=0, p2=0))

    def _set_number_of_evaluations(self, n):
        self.fake_results = types.SimpleNamespace(number_of_evaluations=0)
        self.study._results = self.fake_results
        self.fake_results.number_of_evaluations = n

    def test_default_goals(self):
        self.assertAlmostEqual(
            self.study._rmse_goal,
            1e-2,
        )
        self.assertAlmostEqual(
            self.study._max_abs_error_goal,
            1e-1,
        )

    def test_setting_both_goals(self):
        new_rmse = 5e-3
        new_max = 2e-3
        self.study.set_error_stopping_criteria(
            rmse_goal=new_rmse,
            max_abs_error_goal=new_max,
        )
        self.assertAlmostEqual(
            self.study._rmse_goal,
            new_rmse,
        )
        self.assertAlmostEqual(
            self.study._max_abs_error_goal,
            new_max,
        )

    def test_setting_one_goal_keeps_other_untouched(self):
        new_rmse = 1e-4
        self.study.set_error_stopping_criteria(
            rmse_goal=new_rmse
        )
        self.assertAlmostEqual(
            self.study._rmse_goal,
            new_rmse,
        )
        self.assertAlmostEqual(
            self.study._max_abs_error_goal,
            1e-1,
        )
    def test_invalid_non_numeric_raises_type_error(self):
        with self.assertRaises(TypeError):
            self.study.set_error_stopping_criteria(rmse_goal="bad")

        with self.assertRaises(TypeError):
            self.study.set_error_stopping_criteria(max_abs_error_goal=[1, 2])

    def test_invalid_non_positive_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.study.set_error_stopping_criteria(rmse_goal=0.0)

        with self.assertRaises(ValueError):
            self.study.set_error_stopping_criteria(max_abs_error_goal=-1e-3)

    def test_set_save_filename(self):
        self.assertEqual(self.study._surrogate_save_filename, None)
        with self.assertRaises(ValueError):
            self.study.set_surrogate_save_filename("my_surrogate_name")
        with self.assertRaises(TypeError):
            self.study.set_surrogate_save_filename(0)
        with self.assertRaises(ValueError):
            self.study.set_surrogate_save_filename("")
        self.study.set_surrogate_save_filename("my_surrogate_name.joblib")
        self.assertEqual(self.study._surrogate_save_filename, "my_surrogate_name.joblib")
        self.assertEqual(self.study.surrogate_save_filename, "my_surrogate_name.joblib")

    def test_set_surrogate_storage_options(self):
        self.assertEqual(self.study._surrogate_storage_best_n_surrogates, 1)
        self.assertIsNone(self.study._surrogate_storage_every_n_batches)
        self.assertEqual(self.study._surrogate_storage_score_metric, "max_error")

        self.study.set_surrogate_storage_options(
            best_n_surrogates=3,
            save_every_n_batches=5,
            score_metric="max_error",
        )

        self.assertEqual(self.study._surrogate_storage_best_n_surrogates, 3)
        self.assertEqual(self.study._surrogate_storage_every_n_batches, 5)
        self.assertEqual(self.study._surrogate_storage_score_metric, "max_error")

        self.study.set_surrogate_storage_options(
            best_n_surrogates=None,
            save_every_n_batches=2,
            score_metric="score",
        )

        self.assertIsNone(self.study._surrogate_storage_best_n_surrogates)
        self.assertEqual(self.study._surrogate_storage_every_n_batches, 2)
        self.assertEqual(self.study._surrogate_storage_score_metric, "score")

    def test_set_surrogate_storage_options_invalid_inputs(self):
        with self.assertRaises(ValueError):
            self.study.set_surrogate_storage_options(
                best_n_surrogates=None,
                save_every_n_batches=None,
            )

        with self.assertRaises(ValueError):
            self.study.set_surrogate_storage_options(score_metric="not_a_metric")

        with self.assertRaises(ValueError):
            self.study.set_surrogate_storage_options(best_n_surrogates=0)

        with self.assertRaises(TypeError):
            self.study.set_surrogate_storage_options(save_every_n_batches="bad")

    @unittest.skipIf(not HAS_PYAPPROX,
                 "pyapprox not installed – skipping pyapprox‑dependent tests")
    def test_user_study_set_test_data_as_study_result(self):
        self.study.set_independent_variable("x", np.linspace(0.0, 1.0, 4))
        self.study.set_target_field_name("y")
        self.study.add_evaluation_set(light_model)
        self.study.set_max_training_samples(1)
        self.study.set_number_of_test_samples(1)
        
        p1 = Parameter("p1", 0.0, 1.0, 0.5)
        p2 = Parameter("p2", -1.0, 1.0, 0.0)
        self.simple_parameters =(p1, p2)
        self.test_study = HaltonStudy(*self.simple_parameters)
        self.test_study.set_number_of_samples(50)
        self.test_study.add_evaluation_set(light_model, self.study.results_synchronizer)

        results = self.test_study.launch()
        self.study.set_test_data(results)
        results = self.study.launch()
        self.assertFalse(os.path.exists("test_samples"))
        self.assertIsInstance(self.study.surrogate, SparseGridAdaptiveSurrogate)

    @unittest.skipIf(not HAS_PYAPPROX,
                 "pyapprox not installed – skipping pyapprox‑dependent tests")
    def test_user_study_set_test_data_as_string(self):
        self.study.set_independent_variable("x", np.linspace(0.0, 1.0, 4))
        self.study.set_target_field_name("y")
        self.study.add_evaluation_set(light_model)
        self.study.set_max_training_samples(1)
        self.study.set_number_of_test_samples(1)
        
        p1 = Parameter("p1", 0.0, 1.0, 0.5)
        p2 = Parameter("p2", -1.0, 1.0, 0.0)
        self.simple_parameters =(p1, p2)
        self.test_study = HaltonStudy(*self.simple_parameters)
        self.test_study.set_number_of_samples(50)
        self.test_study.add_evaluation_set(light_model, self.study.results_synchronizer)

        results = self.test_study.launch()
        self.study.set_test_data("final_results.joblib")
        results = self.study.launch()
        self.assertFalse(os.path.exists("test_samples"))
        self.assertIsInstance(self.study.surrogate, SparseGridAdaptiveSurrogate)
    # -------------------------
    # Validation / API errors
    # -------------------------

    @unittest.skipIf(
        not HAS_PYAPPROX,
        "pyapprox not installed – skipping pyapprox‑dependent tests",
    )
    def test_set_sparse_grid_basis_rejects_invalid_basis_type(self):
        sg_study = SparseGridAdaptiveSurrogateStudy(a, b, c)
        with self.assertRaises(ValueError):
            sg_study.set_sparse_grid_basis(basis_type="not-a-basis")

    @unittest.skipIf(
        not HAS_PYAPPROX,
        "pyapprox not installed – skipping pyapprox‑dependent tests",
    )
    def test_set_sparse_grid_basis_rejects_invalid_piecewise_degree(self):
        sg_study = SparseGridAdaptiveSurrogateStudy(a, b, c)
        with self.assertRaises(ValueError):
            sg_study.set_sparse_grid_basis(basis_type="piecewise", piecewise_degree=0)
        with self.assertRaises(ValueError):
            sg_study.set_sparse_grid_basis(basis_type="piecewise", piecewise_degree=4)

    @unittest.skipIf(
        not HAS_PYAPPROX,
        "pyapprox not installed – skipping pyapprox‑dependent tests",
    )
    def test_set_sparse_grid_adaptivity_limits_validation(self):
        sg_study = SparseGridAdaptiveSurrogateStudy(a, b, c)
        # If you did not implement this method, remove/skip this test
        if not hasattr(sg_study, "set_sparse_grid_adaptivity_limits"):
            self.skipTest("set_sparse_grid_adaptivity_limits not implemented")

        with self.assertRaises((ValueError, TypeError)):
            sg_study.set_sparse_grid_adaptivity_limits(max_level=0, pnorm=1.0)
        with self.assertRaises((ValueError, TypeError)):
            sg_study.set_sparse_grid_adaptivity_limits(max_level=2, pnorm=0.0)

    @unittest.skipIf(
        not HAS_PYAPPROX,
        "pyapprox not installed – skipping pyapprox‑dependent tests",
    )
    def test_set_sparse_grid_basis_sets_lagrange(self):
        sg_study = SparseGridAdaptiveSurrogateStudy(a, b, c)
        sg_study.set_sparse_grid_basis(basis_type="lagrange")
        self.assertEqual(sg_study._sg_basis_type, "lagrange")

    @unittest.skipIf(
        not HAS_PYAPPROX,
        "pyapprox not installed – skipping pyapprox‑dependent tests",
    )
    def test_set_sparse_grid_basis_sets_piecewise_degree(self):
        sg_study = SparseGridAdaptiveSurrogateStudy(a, b, c)
        sg_study.set_sparse_grid_basis(basis_type="piecewise", piecewise_degree=2)
        self.assertEqual(sg_study._sg_basis_type, "piecewise")
        self.assertEqual(sg_study._sg_piecewise_degree, 2)

    @unittest.skipIf(
        not HAS_PYAPPROX,
        "pyapprox not installed – skipping pyapprox‑dependent tests",
    )
    def test_set_sparse_grid_adaptivity_limits_sets_values(self):
        sg_study = SparseGridAdaptiveSurrogateStudy(a, b, c)
        if not hasattr(sg_study, "set_sparse_grid_adaptivity_limits"):
            self.skipTest("set_sparse_grid_adaptivity_limits not implemented")

        sg_study.set_sparse_grid_adaptivity_limits(max_level=7, pnorm=2.0)
        self.assertEqual(sg_study._sg_max_level, 7)
        self.assertEqual(sg_study._sg_pnorm, 2.0)

    def test_results_synchronizer_property_before_and_after_add_evaluation_set(self):
        self.assertIsNone(self.study.results_synchronizer)

        self.study.set_independent_variable("x", [0.0, 1.0])
        self.study.set_target_field_name("y")
        self.study.add_evaluation_set(light_model)

        self.assertIsInstance(self.study.results_synchronizer, SimulationResultsSynchronizer)

    def test_set_test_data_invalid_type_raises(self):
        with self.assertRaises(TypeError):
            self.study.set_test_data(123)

    def test_set_test_data_string_invalid_loaded_object_raises(self):
        with open("not_a_study_results.joblib", "wb") as f:
            import pickle
            pickle.dump({"not": "study_results"}, f)

        with self.assertRaises(RuntimeError):
            self.study.set_test_data("not_a_study_results.joblib")

    def test_default_save_filename_is_none_before_launch(self):
        self.assertIsNone(self.study._surrogate_save_filename)
        self.assertIsNone(self.study.surrogate_save_filename)


class ConstantSurrogate:
    """
    Callable surrogate model used in the tests.

    It expects a NumPy array with shape (n_parameters, n_samples) and
    returns an array of shape (n_samples, n_qois).  For simplicity we set
    n_qois == 1 and return a constant value (broadcast to the output shape).  
    This deterministic
    behavior makes it easy to compute expected errors.
    """
    def __init__(self, n_parameters: int, constant: float = 0.0):
        self.n_parameters = n_parameters
        self.constant = constant
        self.surrogate = self
        
    def __call__(self, param_array: np.ndarray) -> np.ndarray:
        n_samples = param_array.shape[1]
        out = np.full((1, n_samples), self.constant, dtype=float)
        return out


class FixedResponseSurrogate:
    """
    Callable surrogate model that returns a fixed response array for scoring.

    The response should be supplied with shape (n_samples, n_qois). The
    SparseGridAdaptiveSurrogate evaluation path expects the underlying surrogate
    function to return shape (n_qois, n_samples), so this class transposes the
    stored response.
    """
    def __init__(self, response):
        self.response = np.asarray(response, dtype=float)
        self.surrogate = self

    def __call__(self, param_array):
        n_samples = param_array.shape[1]
        return self.response[:n_samples, :].T


class TestSparseGridAdaptiveSurrogate(MatcalUnitTest):
    def setUp(self):
        super().setUp(__file__)   

    def _make_surrogate(self, store_all=True, **storage_options):
        # 2 parameters → simple 2-D problem
        param_names = ["p1", "p2"]

        # Test parameters – shape (n_test_samples, n_params)
        test_params = np.array([[0.0, 0.0],
                                [1.0, 1.0]])

        # Test responses – shape (n_test_samples, n_qois)
        test_responses = np.array([[1.0], [2.0]])

        # Most existing tests were written assuming every surrogate object was
        # retained. Keep that behavior for those tests by default. New tests can
        # pass store_all=False to exercise the new default/best-N behavior.
        if store_all:
            storage_options.setdefault("storage_best_n_surrogates", None)
            storage_options.setdefault("storage_every_n_batches", 1)

        surrogate = SparseGridAdaptiveSurrogate(
            target_field_name="target",
            indep_variable_name="independent",
            indep_variable_values=np.array([0.0]),
            test_params=test_params,
            test_responses=test_responses,
            param_names=param_names,
            bounds=np.array([[0.0, 1.0], [0.0, 1.0]]),
            **storage_options,
        )
        return surrogate

    def test_initialization(self):
        surrogate = self._make_surrogate()
        self.assertEqual(len(surrogate._surrogates), 0)
        self.assertEqual(len(surrogate.stored_surrogates), 0)
        self.assertEqual(surrogate.surrogate_records, [])
        self.assertEqual(surrogate._root_mean_squared_errors, [])
        self.assertEqual(surrogate._max_errors, [])
        self.assertEqual(surrogate._sample_counts, [])
        self.assertIsNone(surrogate.current_surrogate)

    def test_add_iteration_computes_errors(self):
        """
        Verify that _add_iteration:
        * stores a deep‑copied surrogate,
        * computes RMSE and max ∞ errors correctly,
        * records the supplied sample count.
        """
        surrogate = self._make_surrogate()

        # Constants surrogate object to be added during the iteration
        sur = ConstantSurrogate(n_parameters=2, constant=0.0)

        # Number of samples that the iteration pretends to have used
        nsamples = 42

        # Run the private method
        surrogate._add_iteration(sur, nsamples)

        # ---- Checks ----------------------------------------------------
        # 1. Surrogate list length increased
        self.assertEqual(len(surrogate._surrogates), 1)

        # 2. The stored surrogate is a *deep* copy (i.e. not the same object)
        self.assertIsNot(surrogate.stored_surrogates[0], sur)

        expected_rmse = np.sqrt(
            np.mean(
                (
                    np.array([[1.0], [2.0]])
                    - np.array([[0.0], [0.0]])
                ) ** 2
            )
        )        
        self.assertAlmostEqual(surrogate._root_mean_squared_errors[0], expected_rmse)

        # 4. Max ∞ error = max(|R_test - 0|) = 2.0
        self.assertAlmostEqual(surrogate._max_errors[0], 2.0)

        # 5. Sample‑count history
        self.assertEqual(surrogate._sample_counts[0], nsamples)

        # 6. Metadata record was created and linked to the retained surrogate.
        self.assertEqual(len(surrogate.surrogate_records), 1)
        self.assertEqual(len(surrogate.stored_surrogate_scores), 1)
        self.assertIn(0, surrogate.stored_surrogates)
        self.assertIn(0, surrogate.stored_surrogate_scores)
        self.assertEqual(surrogate.stored_surrogate_scores[0]["sample_count"], nsamples)

    def test_property_getters(self):
        """current_surrogate, rmse_history, max_error_history, sample_count_history."""
        surrogate = self._make_surrogate()

        sur = ConstantSurrogate(n_parameters=2, constant=0.0)
        surrogate._add_iteration(sur, nsamples=5)

        # current_surrogate should return the stored copy
        self.assertIsInstance(surrogate.current_surrogate, ConstantSurrogate)

        # History properties should match the internal lists
        self.assertEqual(surrogate.rmse_history, surrogate._root_mean_squared_errors)
        self.assertEqual(surrogate.max_error_history, surrogate._max_errors)
        self.assertEqual(surrogate.sample_count_history, surrogate._sample_counts)

    def test_call_with_positional_args(self):
        """
        Call the surrogate with a tuple of positional arguments.
        The surrogate returns a constant value; we verify the output shape
        and that the correct surrogate (index –1) is used.
        """
        # Surrogate constant = 7.5 → expected output array = [[7.5]]
        surrogate = self._make_surrogate()
        surrogate._add_iteration(ConstantSurrogate(n_parameters=2, constant=7.5), 10)

        # The underlying stored surrogate is the one we just added 
        out = surrogate(0.2, 0.8)  

        # Expected shape (n_qois=1, n_samples=1)
        self.assertIsInstance(out, dict)
        self.assertEqual(out["target"].shape, (1, ))
        self.assertAlmostEqual(out["target"][0], 7.5)
        self.assert_close_arrays(out["independent"], np.array([0.0]))

    def test_call_with_keyword_args(self):
        """Same as the positional test but using **kwargs."""
        surrogate = self._make_surrogate()
        surrogate._add_iteration(ConstantSurrogate(n_parameters=2, constant=-3), 10)

        out = surrogate(p2=0.5, p1=0.5)  

        self.assertIsInstance(out, dict)
        self.assertEqual(out["target"].shape, (1, ))
        self.assertAlmostEqual(out["target"][0], -3.0)

    def test_call_batch_evaluate(self):
        """
        When batch_evaluate=True the surrogate receives the raw argument list.
        We pass a (n_parameters, n_samples) array and check the returned
        shape matches (n_qois, n_samples).
        """
        surrogate = self._make_surrogate()
        sur = ConstantSurrogate(2, 2.0)
        surrogate._add_iteration(sur, 1)
        # Create a batch of three samples, each with 2 parameters
        # call array should be n_samplesXn_params
        batch = np.array([[0.0, 0.5, 1.0],   # p1 values
                          [0.0, 0.5, 1.0]]).T  # p2 values

        out = surrogate(batch, batch_evaluate=True)

        self.assertEqual(out["target"].shape, (3, 1))
        self.assertTrue(np.allclose(out["target"], 2.0))

    def test_call_error_on_invalid_input(self):
        """
        Supplying a mismatched number of positional arguments (or an
        incomplete keyword dict) must raise RuntimeError.
        """
        surrogate = self._make_surrogate()
        surrogate._add_iteration(ConstantSurrogate(2, 3), 1)

        # Wrong number of positional arguments (only one while two are required)
        with self.assertRaises(RuntimeError):
            surrogate(0.1)  # missing second parameter

        # Incomplete keyword dict (missing 'p2')
        with self.assertRaises(RuntimeError):
            surrogate(p1=0.2)

        # Incorrect keyword dict 'q2' != 'p2')
        with self.assertRaises(RuntimeError):
            surrogate(p1=0.2, q2=0.1)

        # Both positional and keyword arguments together – also invalid
        with self.assertRaises(RuntimeError):
            surrogate(0.1, p2=0.3)

        #outside of bound (0,1) for both params
        with self.assertRaises(RuntimeError):
            surrogate(-1, 0)
        with self.assertRaises(RuntimeError):
            surrogate(0, 2)

        with self.assertRaises(RuntimeError):
            surrogate([[0, 2], [1, 0]], batch_evaluate=True)
        #verify it takes in kwargs with lists of param values
        res = surrogate(p1=[0, 1], p2=[1, 0])
        self.assertEqual(res["target"].shape, (2,))
        self.assertEqual(res["target"][0], 3.0)
        self.assertEqual(res["target"][1], 3.0)

    def test_call_with_explicit_surrogate_index(self):
        """
        Verify that the surrogate_index argument correctly selects retained
        surrogates, and that the default call evaluates the best surrogate.
        """
        surrogate = self._make_surrogate()

        first = ConstantSurrogate(n_parameters=2, constant=0.0)
        surrogate._add_iteration(first, 1)

        second = ConstantSurrogate(n_parameters=2, constant=9.9)
        surrogate._add_iteration(second, 2)

        # Index 0 should return the first retained surrogate.
        out0 = surrogate(0.0, 0.0, surrogate_index=0)
        self.assertAlmostEqual(out0["target"][0], 0.0)

        # Explicit latest should return the most recent retained surrogate.
        out_latest = surrogate(0.0, 0.0, surrogate_index="latest")
        self.assertAlmostEqual(out_latest["target"][0], 9.9)

        # Positional -1 should also return the last retained surrogate.
        out_last = surrogate(0.0, 0.0, surrogate_index=-1)
        self.assertAlmostEqual(out_last["target"][0], 9.9)

        # Default call should evaluate the best surrogate, not the latest.
        # For test responses [[1], [2]], constant 0.0 is better than 9.9.
        out_default = surrogate(0.0, 0.0)
        self.assertAlmostEqual(out_default["target"][0], 0.0)

        out_best = surrogate(0.0, 0.0, surrogate_index="best")
        self.assertAlmostEqual(out_best["target"][0], 0.0)    

    def test_call_defaults_to_best_surrogate_not_latest(self):
        """
        Verify that calling the adaptive surrogate without surrogate_index uses
        the best retained surrogate, not the most recent retained surrogate.
        """
        surrogate = self._make_surrogate()

        # Test responses are [[1.0], [2.0]].
        #
        # Iteration 0: constant 10.0 -> poor
        # Iteration 1: constant 1.5  -> best
        # Iteration 2: constant 8.0  -> latest, but not best
        surrogate._add_iteration(ConstantSurrogate(2, constant=10.0), nsamples=10)
        surrogate._add_iteration(ConstantSurrogate(2, constant=1.5), nsamples=20)
        surrogate._add_iteration(ConstantSurrogate(2, constant=8.0), nsamples=30)

        self.assertEqual(set(surrogate.stored_surrogates.keys()), {0, 1, 2})
        self.assertEqual(surrogate.best_surrogate_iteration_index, 1)

        default_result = surrogate(0.0, 0.0)
        best_result = surrogate(0.0, 0.0, surrogate_index="best")
        latest_result = surrogate(0.0, 0.0, surrogate_index="latest")
        last_positional_result = surrogate(0.0, 0.0, surrogate_index=-1)

        self.assertAlmostEqual(default_result["target"][0], 1.5)
        self.assertAlmostEqual(best_result["target"][0], 1.5)
        self.assertAlmostEqual(latest_result["target"][0], 8.0)
        self.assertAlmostEqual(last_positional_result["target"][0], 8.0)
    # ------------------------------------------------------------------
    # 9. Full workflow: add several iterations and query histories -----
    # ------------------------------------------------------------------
    def test_multiple_iterations_history(self):
        """
        Run three iterations with different dummy surrogates and verify that
        the histories grow as expected and store the correct error values.
        """
        surrogate = self._make_surrogate()

        # First iteration – constant 0 → error = test_responses
        surrogate._add_iteration(ConstantSurrogate(2, constant=0.0), nsamples=10)

        # Second iteration – constant 1 → error = test_responses - 1
        surrogate._add_iteration(ConstantSurrogate(2, constant=1.0), nsamples=20)

        # Third iteration – constant 2 → error = test_responses - 2
        surrogate._add_iteration(ConstantSurrogate(2, constant=2.0), nsamples=30)

        # Histories should have length 3
        self.assertEqual(len(surrogate.rmse_history), 3)
        self.assertEqual(len(surrogate.max_error_history), 3)
        self.assertEqual(len(surrogate.sample_count_history), 3)

        # Verify sample‑count history matches the values we passed
        self.assertEqual(surrogate.sample_count_history, [10, 20, 30])

        # Compute expected RMSE scores manually for sanity check
        R_test = surrogate._test_responses  # shape (2, 1)
        # Helper to compute error for a given constant
        def expected_rmse(const):
            diff = R_test - const
            return np.sqrt(np.mean(diff ** 2))

        # Expected values
        exp0 = expected_rmse(0.0)
        exp1 = expected_rmse(1.0)
        exp2 = expected_rmse(2.0)

        self.assertAlmostEqual(surrogate.rmse_history[0], exp0)
        self.assertAlmostEqual(surrogate.rmse_history[1], exp1)
        self.assertAlmostEqual(surrogate.rmse_history[2], exp2)

        # Max errors should be max absolute difference
        self.assertAlmostEqual(surrogate.max_error_history[0], np.max(np.abs(R_test - 0.0)))
        self.assertAlmostEqual(surrogate.max_error_history[1], np.max(np.abs(R_test - 1.0)))
        self.assertAlmostEqual(surrogate.max_error_history[2], np.max(np.abs(R_test - 2.0)))

        self.assertEqual(len(surrogate.surrogate_records), 3)
        self.assertEqual(len(surrogate.stored_surrogates), 3)
        self.assertEqual(len(surrogate.stored_surrogate_scores), 3)

        for idx, record in surrogate.stored_surrogate_scores.items():
            self.assertEqual(record["iteration_index"], idx)
            self.assertTrue(record["surrogate_stored"])
            self.assertEqual(record["sample_count"], surrogate.sample_count_history[idx])

    def test_score_returns_nan_for_single_qoi(self):
        surrogate = self._make_surrogate()
        surrogate._add_iteration(ConstantSurrogate(n_parameters=2, constant=0.0), nsamples=5)
        self.assertTrue(np.isnan(surrogate.score()))

    def test_current_surrogate_tracks_latest_iteration(self):
        surrogate = self._make_surrogate()

        first = ConstantSurrogate(n_parameters=2, constant=1.0)
        second = ConstantSurrogate(n_parameters=2, constant=2.0)

        surrogate._add_iteration(first, nsamples=5)
        first_current = surrogate.current_surrogate
        self.assertAlmostEqual(first_current.constant, 1.0)

        surrogate._add_iteration(second, nsamples=10)
        second_current = surrogate.current_surrogate
        self.assertAlmostEqual(second_current.constant, 2.0)

    def test_default_storage_keeps_only_best_surrogate_but_all_scores(self):
        surrogate = self._make_surrogate(store_all=False)

        surrogate._add_iteration(ConstantSurrogate(2, constant=10.0), nsamples=1)
        surrogate._add_iteration(ConstantSurrogate(2, constant=0.0), nsamples=2)
        surrogate._add_iteration(ConstantSurrogate(2, constant=1.5), nsamples=3)

        self.assertEqual(len(surrogate.rmse_history), 3)
        self.assertEqual(len(surrogate.max_error_history), 3)
        self.assertEqual(len(surrogate.sample_count_history), 3)
        self.assertEqual(len(surrogate.surrogate_records), 3)

        # Default behavior is best_n_surrogates=1 by maximum absolute error.
        self.assertEqual(len(surrogate.stored_surrogates), 1)

        stored_index = list(surrogate.stored_surrogates.keys())[0]
        self.assertEqual(stored_index, 2)

        stored_record = surrogate.stored_surrogate_scores[stored_index]
        self.assertTrue(stored_record["surrogate_stored"])
        self.assertIn("best", stored_record["storage_reason"])
        self.assertEqual(stored_record["iteration_index"], stored_index)
        self.assertEqual(stored_record["batch_number"], 3)
        self.assertEqual(stored_record["sample_count"], 3)

        self.assertEqual(surrogate.best_surrogate_iteration_index, stored_index)
        self.assertIs(surrogate.best_surrogate, surrogate.stored_surrogates[stored_index])

        self.assertIsNotNone(surrogate.test_params)
        self.assertIsNotNone(surrogate.test_responses)
        self.assert_close_arrays(surrogate.test_params, np.array([[0.0, 0.0],
                                                                  [1.0, 1.0]]))
        self.assert_close_arrays(surrogate.test_responses, np.array([[1.0],
                                                                     [2.0]]))

    def test_best_n_storage_keeps_best_n_surrogates_and_links_scores(self):
        surrogate = self._make_surrogate(
            store_all=False,
            storage_best_n_surrogates=2,
            storage_every_n_batches=None,
            storage_score_metric="rmse",
        )

        surrogate._add_iteration(ConstantSurrogate(2, constant=10.0), nsamples=1)
        surrogate._add_iteration(ConstantSurrogate(2, constant=0.0), nsamples=2)
        surrogate._add_iteration(ConstantSurrogate(2, constant=1.5), nsamples=3)
        surrogate._add_iteration(ConstantSurrogate(2, constant=1.0), nsamples=4)

        self.assertEqual(len(surrogate.surrogate_records), 4)
        self.assertEqual(len(surrogate.rmse_history), 4)
        self.assertEqual(len(surrogate.stored_surrogates), 2)

        # For test responses [[1], [2]], constants 1.5 and 1.0 are the best two.
        self.assertEqual(set(surrogate.stored_surrogates.keys()), {2, 3})
        self.assertEqual(set(surrogate.stored_surrogate_scores.keys()), {2, 3})

        for idx, record in surrogate.stored_surrogate_scores.items():
            self.assertEqual(record["iteration_index"], idx)
            self.assertTrue(record["surrogate_stored"])
            self.assertIn("best", record["storage_reason"])

    def test_periodic_storage_keeps_every_nth_batch(self):
        surrogate = self._make_surrogate(
            store_all=False,
            storage_best_n_surrogates=None,
            storage_every_n_batches=2,
        )

        surrogate._add_iteration(ConstantSurrogate(2, constant=0.0), nsamples=1)
        surrogate._add_iteration(ConstantSurrogate(2, constant=0.0), nsamples=2)
        surrogate._add_iteration(ConstantSurrogate(2, constant=0.0), nsamples=3)
        surrogate._add_iteration(ConstantSurrogate(2, constant=0.0), nsamples=4)

        self.assertEqual(len(surrogate.surrogate_records), 4)
        self.assertEqual(len(surrogate.rmse_history), 4)

        # Batches 2 and 4 correspond to zero-based iteration indices 1 and 3.
        self.assertEqual(list(surrogate.stored_surrogates.keys()), [1, 3])
        self.assertEqual(list(surrogate.stored_surrogate_scores.keys()), [1, 3])

        for idx, record in surrogate.stored_surrogate_scores.items():
            self.assertEqual(record["iteration_index"], idx)
            self.assertTrue(record["surrogate_stored"])
            self.assertIn("periodic", record["storage_reason"])

    def test_combined_best_and_periodic_storage(self):
        surrogate = self._make_surrogate(
            store_all=False,
            storage_best_n_surrogates=1,
            storage_every_n_batches=2,
            storage_score_metric="rmse",
        )

        surrogate._add_iteration(ConstantSurrogate(2, constant=10.0), nsamples=1)
        surrogate._add_iteration(ConstantSurrogate(2, constant=0.0), nsamples=2)
        surrogate._add_iteration(ConstantSurrogate(2, constant=1.5), nsamples=3)
        surrogate._add_iteration(ConstantSurrogate(2, constant=8.0), nsamples=4)

        # Best is iteration 2. Periodic batches are iterations 1 and 3.
        self.assertEqual(set(surrogate.stored_surrogates.keys()), {1, 2, 3})
        self.assertEqual(set(surrogate.stored_surrogate_scores.keys()), {1, 2, 3})

        self.assertIn("periodic", surrogate.stored_surrogate_scores[1]["storage_reason"])
        self.assertIn("best", surrogate.stored_surrogate_scores[2]["storage_reason"])
        self.assertIn("periodic", surrogate.stored_surrogate_scores[3]["storage_reason"])

    def test_r2_storage_metric_keeps_highest_r2_surrogate(self):
        surrogate = self._make_surrogate(
            store_all=False,
            storage_best_n_surrogates=1,
            storage_every_n_batches=None,
            storage_score_metric="r2",
        )

        # Single QoI gives nan R2 in the current test fixture, so use a two-QoI
        # response to make R2 meaningful.
        surrogate._test_responses = np.array([[1.0, 2.0],
                                              [2.0, 4.0]])

        class LinearResponseSurrogate:
            def __init__(self, scale):
                self.scale = scale
                self.surrogate = self

            def __call__(self, param_array):
                n_samples = param_array.shape[1]
                base = np.array([[1.0, 2.0],
                                 [2.0, 4.0]])
                return self.scale * base[:n_samples, :].T

        surrogate._add_iteration(LinearResponseSurrogate(scale=0.0), nsamples=1)
        surrogate._add_iteration(LinearResponseSurrogate(scale=1.0), nsamples=2)
        surrogate._add_iteration(LinearResponseSurrogate(scale=0.5), nsamples=3)

        self.assertEqual(len(surrogate.stored_surrogates), 1)
        self.assertEqual(list(surrogate.stored_surrogates.keys()), [1])
        self.assertEqual(surrogate.best_surrogate_iteration_index, 1)
        self.assertIn("best", surrogate.stored_surrogate_scores[1]["storage_reason"])

    def test_select_surrogate_by_best_latest_iteration_and_position(self):
        surrogate = self._make_surrogate(
            store_all=False,
            storage_best_n_surrogates=1,
            storage_every_n_batches=2,
            storage_score_metric="rmse",
        )

        surrogate._add_iteration(ConstantSurrogate(2, constant=10.0), nsamples=1)
        surrogate._add_iteration(ConstantSurrogate(2, constant=0.0), nsamples=2)
        surrogate._add_iteration(ConstantSurrogate(2, constant=1.5), nsamples=3)
        surrogate._add_iteration(ConstantSurrogate(2, constant=8.0), nsamples=4)

        # Retained iteration indices are 1 and 3 from periodic retention,
        # plus 2 from best retention.
        self.assertEqual(set(surrogate.stored_surrogates.keys()), {1, 2, 3})

        best_result = surrogate(0.0, 0.0, surrogate_index="best")
        self.assertAlmostEqual(best_result["target"][0], 1.5)

        latest_result = surrogate(0.0, 0.0, surrogate_index="latest")
        self.assertAlmostEqual(latest_result["target"][0], 8.0)

        iteration_result = surrogate(0.0, 0.0, surrogate_index=2)
        self.assertAlmostEqual(iteration_result["target"][0], 1.5)

        positional_result = surrogate(0.0, 0.0, surrogate_index=-1)
        self.assertAlmostEqual(positional_result["target"][0], 8.0)

    def test_storage_options_can_be_updated_on_adaptive_surrogate(self):
        surrogate = self._make_surrogate(store_all=False)

        self.assertEqual(surrogate._storage_best_n_surrogates, 1)
        self.assertIsNone(surrogate._storage_every_n_batches)
        self.assertEqual(surrogate._storage_score_metric, "max_error")

        surrogate.set_surrogate_storage_options(
            best_n_surrogates=3,
            save_every_n_batches=4,
            score_metric="rmse",
        )

        self.assertEqual(surrogate._storage_best_n_surrogates, 3)
        self.assertEqual(surrogate._storage_every_n_batches, 4)
        self.assertEqual(surrogate._storage_score_metric, "rmse")

    def test_storage_options_invalid_on_adaptive_surrogate(self):
        surrogate = self._make_surrogate(store_all=False)

        with self.assertRaises(ValueError):
            surrogate.set_surrogate_storage_options(
                best_n_surrogates=None,
                save_every_n_batches=None,
            )

        with self.assertRaises(ValueError):
            surrogate.set_surrogate_storage_options(score_metric="bad_metric")

        with self.assertRaises(ValueError):
            surrogate.set_surrogate_storage_options(best_n_surrogates=0)

        with self.assertRaises(TypeError):
            surrogate.set_surrogate_storage_options(save_every_n_batches="bad")

    def test_multiple_iterations_history_when_not_all_surrogates_are_stored(self):
        """
        Run multiple adaptive iterations while retaining only the best surrogate.

        This verifies that:
        * all score histories are retained,
        * all sample counts are retained,
        * all per-batch metadata records are retained,
        * only the best surrogate object is retained, and
        * retained surrogate score records correctly link to retained surrogate objects.
        """
        surrogate = self._make_surrogate(store_all=False)

        # Test responses are [[1.0], [2.0]] in _make_surrogate.
        #
        # Iteration 0: constant 10.0 -> very poor
        # Iteration 1: constant 0.0  -> poor
        # Iteration 2: constant 1.5  -> best for responses [1, 2]
        surrogate._add_iteration(ConstantSurrogate(2, constant=10.0), nsamples=10)
        surrogate._add_iteration(ConstantSurrogate(2, constant=0.0), nsamples=20)
        surrogate._add_iteration(ConstantSurrogate(2, constant=1.5), nsamples=30)

        # All histories are retained even though not all surrogate objects are retained.
        self.assertEqual(len(surrogate.rmse_history), 3)
        self.assertEqual(len(surrogate.max_error_history), 3)
        self.assertEqual(len(surrogate.sample_count_history), 3)
        self.assertEqual(len(surrogate.surrogate_records), 3)

        self.assertEqual(surrogate.sample_count_history, [10, 20, 30])

        # Default storage policy is best_n_surrogates=1 using maximum absolute error.
        self.assertEqual(len(surrogate.stored_surrogates), 1)
        self.assertEqual(len(surrogate.stored_surrogate_scores), 1)

        # Compute expected RMSE and max-error histories.
        test_responses = np.array([[1.0], [2.0]])

        constants = [10.0, 0.0, 1.5]
        expected_rmse = [
            np.sqrt(np.mean((test_responses - constant) ** 2))
            for constant in constants
        ]
        expected_max_error = [
            np.max(np.abs(test_responses - constant))
            for constant in constants
        ]

        for idx in range(3):
            self.assertAlmostEqual(surrogate.rmse_history[idx], expected_rmse[idx])
            self.assertAlmostEqual(surrogate.max_error_history[idx], expected_max_error[idx])

        # The best surrogate is iteration 2, corresponding to constant=1.5.
        best_iteration_index = 2
        self.assertEqual(surrogate.best_surrogate_iteration_index, best_iteration_index)
        self.assertEqual(list(surrogate.stored_surrogates.keys()), [best_iteration_index])
        self.assertEqual(list(surrogate.stored_surrogate_scores.keys()), [best_iteration_index])

        # Stored surrogate score record links cleanly to the retained surrogate.
        stored_record = surrogate.stored_surrogate_scores[best_iteration_index]
        self.assertEqual(stored_record["iteration_index"], best_iteration_index)
        self.assertEqual(stored_record["batch_number"], 3)
        self.assertEqual(stored_record["sample_count"], 30)
        self.assertAlmostEqual(stored_record["rmse"], expected_rmse[best_iteration_index])
        self.assertAlmostEqual(stored_record["max_error"], expected_max_error[best_iteration_index])
        self.assertTrue(stored_record["surrogate_stored"])
        self.assertIn("best", stored_record["storage_reason"])

        # Non-retained records are still present but are marked as not stored.
        self.assertFalse(surrogate.surrogate_records[0]["surrogate_stored"])
        self.assertFalse(surrogate.surrogate_records[1]["surrogate_stored"])
        self.assertTrue(surrogate.surrogate_records[2]["surrogate_stored"])

        # The retained object is the best surrogate object.
        self.assertIs(
            surrogate.best_surrogate,
            surrogate.stored_surrogates[best_iteration_index],
        )

        # Evaluating with surrogate_index="best" should use the retained best surrogate.
        prediction = surrogate(0.0, 0.0, surrogate_index="best")
        self.assertIn("target", prediction)
        self.assertAlmostEqual(prediction["target"][0], 1.5)

    def test_default_storage_metric_is_max_error_not_rmse(self):
        """
        Verify that the default best-surrogate selection uses max_error, not RMSE.

        The two candidate surrogates are chosen so that:

        * iteration 0 has lower RMSE but larger max error;
        * iteration 1 has higher RMSE but smaller max error.

        With the desired default metric of max_error, iteration 1 should be
        retained as the best surrogate.
        """
        surrogate = self._make_surrogate(store_all=False)

        # Test responses from _make_surrogate are [[1.0], [2.0]].
        #
        # Iteration 0 prediction: [[1.0], [0.6]]
        # Errors: [[0.0], [1.4]]
        # RMSE ~= 0.9899, max_error = 1.4
        #
        # Iteration 1 prediction: [[0.0], [1.0]]
        # Errors: [[1.0], [1.0]]
        # RMSE = 1.0, max_error = 1.0
        #
        # RMSE would select iteration 0.
        # max_error should select iteration 1.
        surrogate._add_iteration(
            FixedResponseSurrogate(np.array([[1.0], [0.6]])),
            nsamples=10,
        )
        surrogate._add_iteration(
            FixedResponseSurrogate(np.array([[0.0], [1.0]])),
            nsamples=20,
        )

        self.assertEqual(surrogate._storage_score_metric, "max_error")
        self.assertEqual(len(surrogate.rmse_history), 2)
        self.assertEqual(len(surrogate.max_error_history), 2)
        self.assertEqual(len(surrogate.surrogate_records), 2)

        # Confirm the metric conflict.
        self.assertLess(surrogate.rmse_history[0], surrogate.rmse_history[1])
        self.assertGreater(surrogate.max_error_history[0], surrogate.max_error_history[1])

        # Since the default metric is max_error, iteration 1 should be retained.
        self.assertEqual(surrogate.best_surrogate_iteration_index, 1)
        self.assertEqual(list(surrogate.stored_surrogates.keys()), [1])
        self.assertEqual(list(surrogate.stored_surrogate_scores.keys()), [1])

        stored_record = surrogate.stored_surrogate_scores[1]
        self.assertTrue(stored_record["surrogate_stored"])
        self.assertIn("best", stored_record["storage_reason"])
        self.assertAlmostEqual(stored_record["max_error"], 1.0)

    def test_plot_surrogate_vs_test_data_formats_labels_and_units(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        surrogate = self._make_surrogate()
        surrogate._add_iteration(ConstantSurrogate(2, constant=1.5), nsamples=10)

        fig, ax = surrogate.plot_surrogate_vs_test_data(
            xlabel="independent_variable",
            ylabel="target_field",
            independent_variable_units="s",
            target_field_units="K",
        )

        self.assertEqual(ax.get_xlabel(), "independent variable (s)")
        self.assertEqual(ax.get_ylabel(), "target field (K)")
        self.assertEqual(len(ax.lines), 4)

        plt.close(fig)

    def test_plot_surrogate_error_vs_independent_variable_mean_absolute(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        surrogate = self._make_surrogate()
        surrogate._add_iteration(ConstantSurrogate(2, constant=1.5), nsamples=10)

        fig, ax = surrogate.plot_surrogate_error_vs_independent_variable(
            error_type="absolute",
            error_statistic="mean",
            independent_variable_units="s",
            target_field_units="K",
        )

        self.assertEqual(ax.get_xlabel(), "independent (s)")
        self.assertEqual(ax.get_ylabel(), "target absolute error (K)")
        self.assertEqual(len(ax.lines), 1)

        plt.close(fig)

    def test_plot_surrogate_error_vs_independent_variable_individual_curves(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        surrogate = self._make_surrogate()
        surrogate._add_iteration(ConstantSurrogate(2, constant=1.5), nsamples=10)

        fig, ax = surrogate.plot_surrogate_error_vs_independent_variable(
            error_type="signed",
            error_statistic=None,
        )

        self.assertEqual(len(ax.lines), surrogate.test_responses.shape[0])

        plt.close(fig)

    def test_plot_surrogate_error_vs_independent_variable_invalid_error_type(self):
        surrogate = self._make_surrogate()
        surrogate._add_iteration(ConstantSurrogate(2, constant=1.5), nsamples=10)

        with self.assertRaises(ValueError):
            surrogate.plot_surrogate_error_vs_independent_variable(
                error_type="bad_error_type",
            )

    def test_plot_error_history_formats_labels_and_units(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        surrogate = self._make_surrogate()
        surrogate._add_iteration(ConstantSurrogate(2, constant=10.0), nsamples=10)
        surrogate._add_iteration(ConstantSurrogate(2, constant=1.5), nsamples=20)

        fig, ax = surrogate.plot_error_history(
            metrics=("rmse", "max_error"),
            error_units="K",
            yscale="log",
        )

        self.assertEqual(ax.get_xlabel(), "number of training samples")
        self.assertEqual(ax.get_ylabel(), "error or score (K)")
        self.assertEqual(ax.get_yscale(), "log")
        self.assertEqual(len(ax.lines), 2)

        plt.close(fig)


class _FakeSurrogate:
    """
    Minimal surrogate object that mimics the public attributes used by the
    stopping‑criterion method.
    """
    def __init__(self):
        self._root_mean_squared_errors = []   # filled by the test
        self._max_errors = []       # filled by the test
        self._r2_scores = []

    @property
    def rmse_history(self):
        return self._root_mean_squared_errors

    @property
    def max_error_history(self):
        return self._max_errors

    def score(self, index=-1):
        return self._r2_scores[index]


class _FakeResults:
    """
    Simple container that only needs the attribute `number_of_evaluations`.
    """
    def __init__(self, n_evals: int = 0):
        self.number_of_evaluations = n_evals


class _TestStudyStoppingCriteria(SparseGridAdaptiveSurrogateStudy):
    """
    Sub‑class that bypasses the heavy initialization performed by the real
    `SparseGridAdaptiveSurrogateStudy`.  All attributes required by the
    `_stopping_criterion_met` method are injected manually in the test
    cases.
    """
    def __init__(self):
        # Do **not** call the parent __init__ (it expects a full ParameterCollection)
        # Initialise only the fields that the stopping‑criterion method inspects.
        self._rmse_goal = 1e-2   # default in the parent class
        self._max_abs_error_goal = 1e-1
        self._surrogate = _FakeSurrogate()
        self._results = _FakeResults()
        self._number_of_test_samples = 10
        self._test_samples_user_set = True
        self.set_max_training_samples()

class TestSparseGridStoppingCriteria(MatcalUnitTest):
    """
    Test suite for `_stopping_criterion_met`.
    """

    def setUp(self):   # pragma: no cover – required by MatcalUnitTest
        super().setUp(__file__)
        self.study = _TestStudyStoppingCriteria()

    def test_no_stop_on_first_batch(self):
        # populate error histories with values *below* the goals – they must be ignored
        self.study._surrogate._root_mean_squared_errors = [0.0]   # below 1e‑2
        self.study._surrogate._max_errors = [0.0]       # below 1e‑1
        self.study._surrogate._r2_scores = [0.0]
        self.study._results.number_of_evaluations = 0   # far below max_training_samples

        should_stop = self.study._stopping_criterion_met(training_batch_number=0)
        self.assertFalse(should_stop, "Stopping should NOT be triggered on the first batch")

    def test_stop_on_rmse(self):
        self.study._surrogate._root_mean_squared_errors = [5e-3]   # < 1e‑2 goal
        self.study._surrogate._max_errors = [0.5]       # > goal (irrelevant)
        self.study._surrogate._r2_scores = [0.5]

        self.study._results.number_of_evaluations = 0

        should_stop = self.study._stopping_criterion_met(training_batch_number=2)
        self.assertTrue(should_stop,
                        "Stopping should be triggered when RMSE ≤ goal after >1 batch")

    def test_stop_on_max_absolute_error(self):
        self.study._surrogate._root_mean_squared_errors = [0.5]   # > goal (doesn't matter)
        self.study._surrogate._max_errors = [5e-2]      # < 1e‑1 goal
        self.study._surrogate._r2_scores = [0.5]

        self.study._results.number_of_evaluations = 0

        should_stop = self.study._stopping_criterion_met(training_batch_number=3)
        self.assertTrue(should_stop,
                        "Stopping should be triggered when max absolute error ≤ goal after >1 batch")

    def test_stop_on_max_training_samples(self):
        # Set error histories to values that would *not* normally trigger a stop
        self.study._surrogate._root_mean_squared_errors = [1.0]   # > goal
        self.study._surrogate._max_errors = [1.0]      # > goal
        self.study._surrogate._r2_scores = [0.5]


        # Simulate that the study has already used more samples than allowed.
        # The parent class stores the limit in `_max_training_samples`; we set it
        # directly on our test instance.
        self.study._max_training_samples = 100
        self.study._results.number_of_evaluations = 101  # exceed the limit

        should_stop = self.study._stopping_criterion_met(training_batch_number=5)
        self.assertTrue(should_stop,
                        "Stopping should be triggered when number_of_evaluations > max_training_samples")

    def test_no_stop_when_all_conditions_fail(self):
        self.study._surrogate._root_mean_squared_errors = [0.2]   # > 1e‑2
        self.study._surrogate._max_errors = [0.3]      # > 1e‑1
        self.study._surrogate._r2_scores = [0.5]
        self.study._max_training_samples = 1000
        self.study._results.number_of_evaluations = 500   # below limit

        should_stop = self.study._stopping_criterion_met(training_batch_number=4)
        self.assertFalse(should_stop,
                         "Stopping should NOT be triggered when no criteria are met")
        

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
            bounds = np.array([[-5, 5] for d in np.arange(dim)])
            X_init, _, _, _, bounds = TestVoronoiTessellation.voronoi_initialization(dim, nsamples, bounds)
            
            for fo in [True, False]:
                vor = VoronoiTessellation(X_init, bounds, finite_only=fo)
                vor.build()
            
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
       
    def test_identify_vertices_outside_bounds(self):
        dims = [2, 3]
        for dim in dims:
            nsamples = 2 ** dim
            bounds = np.array([[-5, 5] for d in np.arange(dim)])
            X_init, _, _, _, bounds = TestVoronoiTessellation.voronoi_initialization(dim, nsamples, bounds)
            vor = VoronoiTessellation(X_init, bounds, finite_only=False)
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
        bounds = np.array([[-5, 5], [-5, 5]])
        X_init, _, _, _, bounds = TestVoronoiTessellation.voronoi_initialization(dim, nsamples, bounds)
        vor = VoronoiTessellation(X_init, bounds, finite_only=False)
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
        bounds = np.array([[-5, 5], [-5, 5], [-5, 5]])
        X_init, _, _, _, bounds = TestVoronoiTessellation.voronoi_initialization(dim, nsamples, bounds)
        vor = VoronoiTessellation(X_init, bounds, finite_only=False)
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
            bounds = np.array([[-5, 5] for d in np.arange(dim)])
            X_init, _, _, _, bounds = TestVoronoiTessellation.voronoi_initialization(dim, nsamples, bounds)
            for fo in [True, False]:
                vor = VoronoiTessellation(X_init, bounds, finite_only=fo)
                vor.build()
            
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
            bounds = np.array([[-5, 5] for d in np.arange(dim)])
            X_init, _, _, _, bounds = TestVoronoiTessellation.voronoi_initialization(dim, nsamples, bounds)
            vor = VoronoiTessellation(X_init, bounds, finite_only=False)
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
        bounds = np.array([[-5, 5] for d in np.arange(dim)])
        X_init, _, _, _, bounds = TestVoronoiTessellation.voronoi_initialization(dim, nsamples, bounds)

        vor = VoronoiTessellation(X_init, bounds, finite_only=False)
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
        bounds = np.array([[-5, 5] for d in np.arange(dim)])
        X_init, _, _, _, bounds = TestVoronoiTessellation.voronoi_initialization(dim, nsamples, bounds)

        vor = VoronoiTessellation(X_init, bounds, finite_only=False)
        vor.build()
                       
        with self.assertRaises(ValueError):
            vor.raise_if_invalid_region_index(50)
         
        with self.assertRaises(ValueError):
            vor.raise_if_invalid_region_index(-1)
            
    def test_replace_unbounded_vertices_error_handling(self):
        dims = [2, 3]
        for dim in dims:
            nsamples = 2 ** dim
            bounds = np.array([[-5, 5] for d in np.arange(dim)])
            X_init, _, _, _, bounds = TestVoronoiTessellation.voronoi_initialization(dim, nsamples, bounds)
            for fo in [True, False]:
                vor = VoronoiTessellation(X_init, bounds, finite_only=fo)
                with self.assertRaises(ValueError):
                    vor.replace_unbounded_vertices([-2, -2, -2, -2], 100, [(1, -2), (2, -3), (3, -2), (4,-2)])
                      
    def test_get_closest_seed(self):
        dims = [2, 3]
        for dim in dims:
            nsamples = 2 ** dim
            bounds = np.array([[-5, 5] for d in np.arange(dim)])
            X_init, _, _, _, bounds = TestVoronoiTessellation.voronoi_initialization(dim, nsamples, bounds)
            vor = VoronoiTessellation(X_init, bounds, finite_only=False)
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
        from scipy.spatial import Delaunay
        
        # Create polyhedron from region vertices
        # sample point from within polygon, and assert that point is in the region
        dims = [2, 3]
        for dim in dims:
            nsamples = 2 ** dim
            bounds = np.array([[-5, 5] for d in np.arange(dim)])
            X_init, _, _, _, bounds = TestVoronoiTessellation.voronoi_initialization(dim, nsamples, bounds)
            vor = VoronoiTessellation(X_init, bounds, finite_only=False)
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
            bounds = np.array([[-5, 5] for d in np.arange(dim)])
            X_init, _, _, _, bounds = TestVoronoiTessellation.voronoi_initialization(dim, nsamples, bounds)
            vor = VoronoiTessellation(X_init, bounds, finite_only=False)
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
            bounds = np.array([[-5, 5] for d in np.arange(dim)])
            X_init, _, _, _, bounds = TestVoronoiTessellation.voronoi_initialization(dim, nsamples, bounds)
            vor = VoronoiTessellation(X_init, bounds, finite_only=False)
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
        self.model = PythonModel(linear_model_2d)
        self.model.set_name('quadratic_2d')
        theta1 = Parameter('a', -5, 5, distribution="uniform_uncertain")
        theta2 = Parameter('b', -5, 5, distribution="uniform_uncertain")
        pc = ParameterCollection("two_parameter", theta1, theta2)
        study = self._study_class(pc)
        self.nsamples = 20
        self.test_points = np.linspace(.25, .75, 10)
        objective = SimulationResultsSynchronizer("x", self.test_points, "f")
        study.add_evaluation_set(self.model, objective)
        study.set_number_of_samples(self.nsamples)
        study_info = study.launch()
        self.X, self.y = TestKFoldCrossValidation.format_study_params_and_output(study_info)
        
    def test_initialization(self):
        self.kfold = KFoldCrossValidation(5, False, 'x', [0,1], None, 'rmse', 
                                         'y', None, {})

        self.assertEqual(self.kfold.nsplits, 5)
        self.assertFalse(self.kfold.group_kfold)
        self.assertIsNone(self.kfold.scale)
        self.assertEqual(self.kfold.metric, 'rmse')
        self.assertEqual(self.kfold.interpolation_field, 'x')
        self.assertIsNone(self.kfold.param_names)

    def test_set_kfcv_options(self):
        kfcv_options = {'nsplits':4, 'group_kfold':True,
                        'scale': 'cbrt', 'metric':'nlpd',
                        'interpolation_field': 'x',
                        'interpolation_values':self.test_points,                         
                        'param_names': ['a', 'b'],
                        'target_field':'f', 
                        'surrogate_options':{}
                        }
        self.kfold = KFoldCrossValidation(**kfcv_options)
        self.assertEqual(self.kfold.nsplits, 4)
        self.assertTrue(self.kfold.group_kfold)
        self.assertEqual(self.kfold.scale, 'cbrt')
        self.assertEqual(self.kfold.metric, 'nlpd')
        self.assertEqual(self.kfold.interpolation_field, 'x')
        self.assertEqual(self.kfold.param_names, ['a', 'b'])
         
        # Test setting splits > number of samples. Should revert to length of X
        kfcv_options = {'nsplits':20, 'group_kfold':True,
                        'scale': 'cbrt', 'metric':'nlpd',
                        'interpolation_field': 'x',
                        'interpolation_values':self.test_points,                         
                        'param_names': ['a', 'b'],
                        'target_field':'f', 
                        'surrogate_options':{}
                        } 
        self.kfold = KFoldCrossValidation(**kfcv_options)
        self.kfold._check_npslits(np.zeros((5,2)))
        self.assertEqual(self.kfold.nsplits, 2)

    def test_group_kfold_cv(self):
        nsplits = 4
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=nsplits, random_state=42)
        groups = kmeans.fit_predict(self.X)
        
        kfcv_options = {'group_kfold': True,
                        'nsplits': nsplits,
                        'scale':1,
                        'metric': 'rmse',
                        'interpolation_field': 'x',
                        'interpolation_values':self.test_points,                         
                        'param_names': ['a', 'b'],
                        'target_field':'f', 
                        'surrogate_options':{}}  
        self.kfold = KFoldCrossValidation(**kfcv_options)
        kf_results = self.kfold.perform_kfold_cv(self.X, self.y, groups)
        self.assertTrue(len(kf_results) == nsplits)
        
    def test_perform_kfold_cv(self):
        kfcv_options = {'nsplits':5, 
                        'scale':1,
                        'group_kfold':False,
                        'metric': 'rmse',
                        'interpolation_field':'x',
                        'interpolation_values':self.test_points,                         
                        'param_names':['a', 'b'],
                        'target_field':'f', 
                        'surrogate_options':{}}
        self.kfold = KFoldCrossValidation(**kfcv_options)
        kf_results = self.kfold.perform_kfold_cv(self.X, self.y, None)
       
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
        kfcv_options = {'nsplits':5, 
                        'group_kfold':False,
                        'scale':1,
                        'metric': 'rmse',
                        'interpolation_field':'x',
                        'interpolation_values':self.test_points,                         
                        'param_names':['a', 'b'], 
                        'target_field':'f', 
                        'surrogate_options':{}}
        self.kfold = KFoldCrossValidation(**kfcv_options)
        self.kfold.X = self.X
        self.kfold.y = self.y
        self.kfold.par_names = ['a', 'b']
        train_index = [0, 1, 2]
        test_index = [3, 4]
        error, test_idx_returned = self.kfold.evaluate_fold(train_index, test_index, self.X, self.y, 0)
        self.assertEqual(test_idx_returned, test_index)
        self.assertIsInstance(error, float)          
       
        
class TestLeaveOneOutCrossValidation(MatcalUnitTest):
    _study_class = HaltonStudy
    
    def setUp(self):
        super().setUp(__file__)
        self.test_points = np.linspace(.25, .75, 10)

    def test_initialization(self):
        loocv_options = {'scale': None, 'metric':'rmse',
                        'interpolation_field': 'x',
                        'interpolation_values':self.test_points, 
                        'par_names': None, 
                        'target_field':'f', 
                        'surrogate_options':{}
                        }
        self.loocv = LeaveOneOutCrossValidation(**loocv_options)
        self.assertIsNone(self.loocv.scale)
        self.assertEqual(self.loocv.metric, 'rmse')
        self.assertEqual(self.loocv.interpolation_field, 'x')
        self.assertIsNone(self.loocv.par_names)

    def test_perform_loocv(self):
        self.model = PythonModel(linear_model_2d)
        self.model.set_name('quadratic_2d')
        theta1 = Parameter('a', -5, 5, distribution="uniform_uncertain")
        theta2 = Parameter('b', -5, 5, distribution="uniform_uncertain")
        study = self._study_class(theta1, theta2)
        self.nsamples = 10
        objective = SimulationResultsSynchronizer("x", self.test_points, "f")
        study.add_evaluation_set(self.model, objective)
        study.set_number_of_samples(self.nsamples)
        study_info = study.launch()
        self.X, self.y = TestKFoldCrossValidation.format_study_params_and_output(study_info)
        indices = range(self.nsamples)
        loocv_options = {'scale': None, 'metric':'nlpd',
                    'interpolation_field': 'x',
                        'interpolation_values':self.test_points, 
                        'par_names': ['a', 'b'],
                        'target_field':'f', 
                        'surrogate_options':{}
                        }
        self.loocv = LeaveOneOutCrossValidation(**loocv_options)
        loo_results = self.loocv.perform_loocv(self.X, self.y, indices)
            
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
            physical_model = PythonModel(linear_model_2d)
            model_name = 'quadratic_2d'
        elif dim == 3:            
            physical_model = PythonModel(quadratic_model_3d)
            model_name = 'quadratic_3d'
        physical_model.set_name(model_name)
        parameter_collection =  TestVoronoiAdaptiveSurrogateStudy.setup_parameter_collection(dim)

        return physical_model, parameter_collection

    def setUp(self):
        super().setUp(__file__)
    
    def setup_study(self, dim):
        physical_model, parameter_collection = TestVoronoiAdaptiveSurrogateStudy.setup_model(dim)
        vor_study = self._study_class(parameter_collection)

        indep_vals = np.linspace(.25, .75, 20)
        vor_study.set_independent_variable("x", indep_vals)
        vor_study.set_target_field_name("f")
        vor_study.add_evaluation_set(physical_model)
        vor_study.set_max_training_samples(5)
        vor_study.set_number_of_initial_samples(10)
        vor_study.set_cross_validation_options(nsplits=5, nmax_loo='all', cv_metric='nlpd')
        return vor_study

    def test_initialization(self):
        dims = [2, 3]
        for dim in dims:
            vor_study = self.setup_study(dim)
            vor_study.launch()
            # build surrogate that given a, b, c gives you f
            self.assertFalse(vor_study._finite_only)
            self.assertTrue(vor_study._iterative_updates)
            self.assertEqual(vor_study._voronoi_type, 'full')
            
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
            surr_score = vor_study._current_surrogate_score['score']
            rmse = vor_study._current_surrogate_score['rmse']
            nlpd = vor_study._current_surrogate_score['nlpd']
            self.assertGreater(len(surr_score), 0)
            self.assertTrue(np.all([i > 0 for i in surr_score]))
            self.assertTrue(np.all([i > 0 for i in rmse]))
            shutil.rmtree("test_samples")
            
    def test_convergence(self):
        vor_study = self.setup_study(2)
        vor_study.set_max_training_samples(50)
        vor_study.set_cross_validation_options(nsplits=0, nmax_loo='all', 
                                               cv_metric='rmse')
        vor_study.set_error_stopping_criteria(1e-8, 1e-8)
        vor_study.set_convergence_criteria(1e-1, 'rmse')
        vor_study.launch()
        score = vor_study._current_surrogate_score
        metric = vor_study._convergence_metric
        converged = np.abs(score[metric][-1] - score[metric][-2]) <= vor_study._eps
        self.assertTrue(converged)
        
    def test_nmax_loo_all(self):
        vor_study = self.setup_study(2)
        vor_study.set_max_training_samples(15)
        vor_study.set_number_of_initial_samples(7)
        vor_study.set_cross_validation_options(nsplits=2, nmax_loo='all')
        vor_study.set_error_stopping_criteria(1e-8, 1e-8)
        vor_study.set_convergence_criteria(1e-1, 'rmse')
        vor_study.launch()
        self.assertIsNone(vor_study._loo_errors)
        
    def test_thin(self):
        vor_study = self.setup_study(2)
        vor_study.set_max_training_samples(7)
        vor_study.set_number_of_initial_samples(6)
        vor_study.set_cross_validation_options(nsplits=0)
        vor_study.set_voronoi_sampling_options(thin=2)
        vor_study.set_seed(100)
        vor_study.set_error_stopping_criteria(1e-8, 1e-8)
        vor_study.set_convergence_criteria(1e-1, 'rmse')
        vor_study.launch()
        self.assertEqual(vor_study._nbatch_samples[-1], 
                         vor_study._num_initial_samples*1.5)

    def test_random_selection(self):
        vor_study = self.setup_study(2)
        vor_study.set_max_training_samples(7)
        vor_study.set_number_of_initial_samples(6)
        vor_study.set_cross_validation_options(nsplits=0)
        vor_study.set_voronoi_sampling_options(random_selection=3)
        vor_study.set_seed(100)
        vor_study.set_error_stopping_criteria(1e-8, 1e-8)
        vor_study.set_convergence_criteria(1e-1, 'rmse')
        vor_study.launch()
        self.assertEqual(vor_study._nbatch_samples[-1], 
                        vor_study._num_initial_samples*1.5)

    def test_local_tess(self):
        vor_study = self.setup_study(2)
        vor_study.set_max_training_samples(21)
        vor_study.set_number_of_initial_samples(20)
        vor_study.set_cross_validation_options(nsplits=0)
        vor_study.set_voronoi_sampling_options(random_selection=3, 
                                               voronoi_type='local')
        vor_study.set_seed(42)
        vor_study.set_error_stopping_criteria(1e-8, 1e-8)
        vor_study.set_convergence_criteria(1e-1, 'rmse')
        vor_study.launch()
        self.assertIsNotNone(vor_study._tree)
        
    def test_group_kfold(self):
        vor_study = self.setup_study(2)
        vor_study.set_max_training_samples(21)
        vor_study.set_number_of_initial_samples(20)
        vor_study.set_cross_validation_options(nsplits=4, group_kfold=True)
        vor_study.set_voronoi_sampling_options(random_selection=10, 
                                               voronoi_type='local')
        vor_study.set_error_stopping_criteria(1e-8, 1e-8)
        vor_study.set_convergence_criteria(1e-1, 'rmse')
        vor_study.launch()
        self.assertTrue(vor_study._nbatch_samples[-1] > vor_study._num_initial_samples)

    def test_perform_cv_and_find_max_errors(self):
        dims = [2, 3]
        for dim in dims:
            vor_study = self.setup_study(dim)           
            nsplits = 2
            nmax_loo = 3
            vor_study.set_max_training_samples(21)
            vor_study.set_number_of_initial_samples(20)
            vor_study.set_cross_validation_options(nsplits=nsplits, cv_metric='nlpd', 
                                                   nmax_loo=3, nmax_folds=1)
            vor_study.set_error_stopping_criteria(1e-8, 1e-8)
            vor_study.set_convergence_criteria(1e-1, 'rmse') 
            vor_study.launch()
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
            self.assertTrue(np.all(np.isin(np.arange(vor_study._num_initial_samples), test_indices)))
            self.assertEqual(len(np.unique(test_indices)), vor_study._num_initial_samples)

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
            training_params = vor_study._format_params(vor_study._results)

            worst_sample_locations = vor_study._find_loo_max_errors(training_params)
            max_error_indices = sorted_array[-nmax_loo:, :][:, 1][::-1] # get indices of largest errors and reverse (greatest to smallest)
            max_error_indices = [int(x) for x in max_error_indices] # convert entries to int
            self.assertTrue(np.all(worst_sample_locations == training_params[max_error_indices]))
            shutil.rmtree("test_samples")

    def test_adaptive_voronoi_surrogate_generation(self):
        #move to integration
        dims = [2, 3]
        for dim in dims:
            vor_study = self.setup_study(dim)
            
            cross_validation_options = {'nsplits':2,
                                        'nmax_folds':1,
                                        'nmax_loo':5,
                                        'cv_metric': 'nlpd'
                                        }
            vor_study.set_max_training_samples(60)
            vor_study.set_number_of_initial_samples(10)
            vor_study.set_convergence_criteria(convergence_metric='score')
            vor_study.set_cross_validation_options(**cross_validation_options)
             
            vor_study.launch()
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
            samples = vor_study._format_params(vor_study._results)
            self.assertEqual(samples.shape[0], vor_study._nbatch_samples[-1])
            
            # verify that all samples are within bounds
            lb = vor_study._bounds[:,0]
            ub = vor_study._bounds[:,1]
            outside_samples = (samples < lb).any(axis=1) | (samples > ub).any(axis=1)
            self.assertFalse(np.any(outside_samples))
            shutil.rmtree("test_samples")

    def test_call_error_on_invalid_input(self):
        """
        Supplying a mismatched number of positional arguments (or an
        incomplete keyword dict) must raise RuntimeError.
        """
        vor_study = self.setup_study(2)
        vor_study.set_max_training_samples(7)
        vor_study.set_number_of_initial_samples(6)
        vor_study.set_cross_validation_options(nsplits=0)
        vor_study.set_voronoi_sampling_options(random_selection=3)
        vor_study.set_seed(100)
        vor_study.set_error_stopping_criteria(1e-8, 1e-8)
        vor_study.set_convergence_criteria(1e-1, 'rmse')
        vor_study.launch()
        surrogate = vor_study._surrogate

        # Wrong number of positional arguments (only one while two are required)
        with self.assertRaises(RuntimeError):
            surrogate(0.1)  # missing second parameter

        # Incomplete keyword dict (missing 'b')
        with self.assertRaises(RuntimeError):
            surrogate(a=0.2)

        # Incorrect keyword dict 'a2' != 'b')
        with self.assertRaises(RuntimeError):
            surrogate(a=0.2, a2=0.1)

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
        res = surrogate(a=[0, 1], b=[1, 0])
        self.assertEqual(res["f"].shape, (2,20))
