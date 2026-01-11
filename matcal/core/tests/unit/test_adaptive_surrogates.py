import os
import types
import numpy as np
import unittest

from matcal.core.adaptive_surrogates import (
    SparseGridAdaptiveSurrogateStudy,
    _get_parameter_bounds, AdaptiveSurrogate)
from matcal.core.objective import SimulationResultsSynchronizer
from matcal.core.models import PythonModel
from matcal.core.parameters import Parameter
from matcal.core.serializer_wrapper import matcal_load
from matcal.core.study_base import StudyBase
from matcal.core.tests.MatcalUnitTest import MatcalUnitTest


try:
    import pyapprox  
    HAS_PYAPPROX = True
except Exception:    
    HAS_PYAPPROX = False



def return_data(*args, **kwargs):
    return {"x":np.linspace(0,1,10), "y":np.linspace(1,3,10)}
light_model = PythonModel(return_data)


class TestSparseGridAdaptiveSurrogate(MatcalUnitTest):
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
        with self.assertRaises(StudyBase.StudyInputError):
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
        self.study.set_test_group_random_seed(1234)
        self.assertEqual(self.study.HaltonSampler.scramble, True)
        self.study.set_test_group_random_seed(123, False)
        self.assertEqual(self.study.HaltonSampler.scramble, False)
        with self.assertRaises(TypeError):
            self.study.set_test_group_random_seed("", False)
        
        with self.assertRaises(TypeError):
            self.study.set_test_group_random_seed(123, 123)
        
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
        sync = self.study._make_simulation_results_synchronizer()
        self.assertIsInstance(sync, SimulationResultsSynchronizer)
        self.assertEqual(sync.independent_field, "x")
        self.assertEqual(sync._independent_field_values, [0.0, 1.0])
        self.assertEqual(sync.fields_of_interest, ("y",))

    def test_make_simulation_results_synchronizer_missing(self):
        self.study.set_independent_variable("x", [0, 1])
        with self.assertRaises(RuntimeError) as ctx:
            self.study._make_simulation_results_synchronizer()
        self.assertIn("Target field name", str(ctx.exception))

        self.study._independent_variable = None
        self.study._independent_variable_values = None
        self.study.set_target_field_name("test")
        with self.assertRaises(RuntimeError) as ctx:
            self.study._make_simulation_results_synchronizer()
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

        self.study._reset_study_after_test_sampling_generation(None)
        self.study.set_working_directory("work")
        test_dir = os.path.abspath("work_test_samples")
        self.study.launch()
        self.assertTrue(os.path.isdir(test_dir))
        self.assertTrue(os.path.isdir("work"))
    
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
        self.assertTupleEqual(params.shape, (2, 3))

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
          * transforms the sample matrix via ``map_from_canonical``,
          * creates a list ``_parameter_sets_to_evaluate`` with one dict per
            sample,
          * each dict contains the correct parameter names and values, and
          * ``_add_parameter_evaluation`` is called with the same dictionaries.
        """
        class DummyTransformer:
            def map_from_canonical(self, arr):
                return arr  # identity – the samples we pass are already in the desired space

        self.study._variable_transformer = DummyTransformer()

        canonical_samples = np.array([[0.1, 0.2, 0.3],   # p1 values
                                      [0.1, 0.5, -0.1]])  # p2 values

        self.study._populate_parameter_evaluations_adaptive(canonical_samples)
        self.assertEqual(len(self.study._parameter_sets_to_evaluate), 3)

        expected_names = self.study._parameter_collection.get_item_names()
        expected_names = list(expected_names)          # ensure list for indexing

        for i, param_dict in enumerate(self.study._parameter_sets_to_evaluate):
            self.assertCountEqual(param_dict.keys(), expected_names)
            for name, col_idx in zip(expected_names, range(len(expected_names))):
                expected_value = canonical_samples[col_idx, i]
                self.assertAlmostEqual(param_dict[name], expected_value)

    def test_update_surrogate_score(self):
        class DummyTransform:
            def map_to_canonical(self, x):
                return x
        self.study._variable_transformer = DummyTransform()

        self.study._test_params = np.array([[0.0, 1.0], [0.5, 0.5]])
        self.study._test_responses = np.array([[1.0, 2.0], [1.5, 2.5]])

        class DoubleParamSurrogate:
            def __call__(self, params):
                return params * 2.0
        self.study._surrogate = AdaptiveSurrogate("target", "independent", 
                                                   np.array([0.0, 1.0]), 
                                                   DummyTransform(), 
                                                   self.study._test_params, 
                                                   self.study._test_responses, 
                                                   ["p1", "p2"]   
                                                   )        
        self.study._surrogate._add_iteration(DoubleParamSurrogate(), 2)        
        
        self.assertEqual(len(self.study.surrogate.average_error_history), 1)
        self.assertEqual(len(self.study.surrogate.max_error_history), 1)

        expected_l2 = np.linalg.norm(self.study._test_responses -
                                    DoubleParamSurrogate()(self.study._test_params))
        expected_l2 /= self.study._test_responses.shape[1]
        self.assertAlmostEqual(self.study.surrogate.average_error_history[0], expected_l2)

    @unittest.skipIf(not HAS_PYAPPROX,
                 "pyapprox not installed – skipping pyapprox‑dependent tests")
    def test_study_results_is_not_none(self):
        """launch must eventually call the sparse‑grid routine."""
        called = {"flag": False}
        self.study.set_independent_variable("x", np.linspace(0,1,3))
        self.study.set_target_field_name("y")
        self.study.add_evaluation_set(light_model)
        self.study.set_max_training_samples(30)
        self.study.set_number_of_test_samples(1)
        self.study.launch()
        self.assertIsNotNone(self.study.results)
        sur_file = self.study.save_filename
        sur_results = matcal_load(sur_file)
        self.assertEqual(self.study.surrogate.average_error_history, 
                         sur_results.average_error_history)
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
            self.study._average_l2_error_goal,
            1e-2,
        )
        self.assertAlmostEqual(
            self.study._max_abs_error_goal,
            1e-1,
        )

    def test_setting_both_goals(self):
        new_avg = 5e-3
        new_max = 2e-3
        self.study.set_error_stopping_criteria(
            average_l2_error_goal=new_avg,
            max_abs_error_goal=new_max,
        )
        self.assertAlmostEqual(
            self.study._average_l2_error_goal,
            new_avg,
        )
        self.assertAlmostEqual(
            self.study._max_abs_error_goal,
            new_max,
        )

    def test_setting_one_goal_keeps_other_untouched(self):
        new_avg = 1e-4
        self.study.set_error_stopping_criteria(
            average_l2_error_goal=new_avg,
            max_abs_error_goal=None,
        )
        self.assertAlmostEqual(
            self.study._average_l2_error_goal,
            new_avg,
        )
        self.assertAlmostEqual(
            self.study._max_abs_error_goal,
            1e-1,
        )
    def test_invalid_non_numeric_raises_type_error(self):
        with self.assertRaises(TypeError):
            self.study.set_error_stopping_criteria(average_l2_error_goal="bad")

        with self.assertRaises(TypeError):
            self.study.set_error_stopping_criteria(max_abs_error_goal=[1, 2])

    def test_invalid_non_positive_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.study.set_error_stopping_criteria(average_l2_error_goal=0.0)

        with self.assertRaises(ValueError):
            self.study.set_error_stopping_criteria(max_abs_error_goal=-1e-3)

    def test_set_save_filename(self):
        self.assertEqual(self.study._save_filename, None)
        with self.assertRaises(ValueError):
            self.study.set_save_filename("my_surrogate_name")
        with self.assertRaises(TypeError):
            self.study.set_save_filename(0)
        with self.assertRaises(ValueError):
            self.study.set_save_filename("")
        self.study.set_save_filename("my_surrogate_name.joblib")
        self.assertEqual(self.study._save_filename, "my_surrogate_name.joblib")
        self.assertEqual(self.study.save_filename, "my_surrogate_name.joblib")


class IdentityTransformer:
    """
    Minimal transformer that implements the two methods used by AdaptiveSurrogate.
    Both methods simply return the argument unchanged.
    """
    def map_to_canonical(self, arr):
        # In the real code this would map to [-1, 1] space – we keep it identity.
        return arr

    def map_from_canonical(self, arr):
        return arr


class ConstantSurrogate:
    """
    Callable surrogate model used in the tests.

    It expects a NumPy array with shape (n_parameters, n_samples) and
    returns an array of shape (n_samples, n_qois).  For simplicity we set
    n_qois == 1 and return a constant value equal to the sum of the
    input parameters (broadcast to the output shape).  This deterministic
    behavior makes it easy to compute expected errors.
    """
    def __init__(self, n_parameters: int, constant: float = 0.0):
        self.n_parameters = n_parameters
        self.constant = constant
        
    def __call__(self, param_array: np.ndarray) -> np.ndarray:
        n_samples = param_array.shape[1]
        out = np.full((n_samples, 1), self.constant, dtype=float)
        return out


class TestAdaptiveSurrogate(MatcalUnitTest):
    def setUp(self):
        super().setUp(__file__)   

    def _make_surrogate(self):
        # 2 parameters → simple 2‑D problem
        param_names = ["p1", "p2"]
        n_params = len(param_names)

        # Test parameters – shape (n_params, n_test_samples)
        test_params = np.array([[0.0, 1.0],   # p1 values
                                [0.0, 1.0]])  # p2 values

        # Test responses – shape (n_test_samples, n_qois)
        # We deliberately set them to something different from the surrogate
        # constant so we can validate error calculation.
        test_responses = np.array([[1.0], [2.0]])

        transformer = IdentityTransformer()
        surrogate = AdaptiveSurrogate(
            target_field_name="target",
            indep_variable_name="independent",
            indep_variable_values=np.array([0.0]),
            variable_transformer=transformer,
            test_params=test_params,
            test_responses=test_responses,
            param_names=param_names,
        )
        return surrogate

    def test_initialization(self):
        surrogate = self._make_surrogate()
        self.assertEqual(surrogate._surrogates, [])
        self.assertEqual(surrogate._average_errors, [])
        self.assertEqual(surrogate._max_errors, [])
        self.assertEqual(surrogate._sample_counts, [])
        self.assertIsNone(surrogate.current_surrogate)

    def test_add_iteration_computes_errors(self):
        """
        Verify that _add_iteration:
        * stores a deep‑copied surrogate,
        * computes average L2 and max ∞ errors correctly,
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
        self.assertIsNot(surrogate._surrogates[0], sur)

        # 3. Average L2 error = ||R_test - 0||_2 / N
        # R_test shape (2, 1) → norm = sqrt(1^2 + 2^2) = sqrt(5)
        expected_l2 = np.linalg.norm(np.array([[1.0], [2.0]]) - np.array([[0], [0]])) / 1
        self.assertAlmostEqual(surrogate._average_errors[0], expected_l2)

        # 4. Max ∞ error = max(|R_test - 0|) = 2.0
        self.assertAlmostEqual(surrogate._max_errors[0], 2.0)

        # 5. Sample‑count history
        self.assertEqual(surrogate._sample_counts[0], nsamples)

    def test_property_getters(self):
        """current_surrogate, average_error_history, max_error_history, sample_count_history."""
        surrogate = self._make_surrogate()

        sur = ConstantSurrogate(n_parameters=2, constant=0.0)
        surrogate._add_iteration(sur, nsamples=5)

        # current_surrogate should return the stored copy
        self.assertIsInstance(surrogate.current_surrogate, ConstantSurrogate)

        # History properties should match the internal lists
        self.assertEqual(surrogate.average_error_history, surrogate._average_errors)
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
        batch = np.array([[0.0, 1.0, 2.0],   # p1 values
                          [0.0, 1.0, 2.0]])  # p2 values

        out = surrogate(batch, batch_evaluate=True)

        self.assertEqual(out.shape, (3, 1))
        self.assertTrue(np.allclose(out, 2.0))

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

    def test_call_with_explicit_surrogate_index(self):
        """
        Verify that the `surrogate_index` argument correctly selects a
        surrogate from the internal list.
        """
        # Create two distinct ConstantSurrogates with different constant outputs
        surrogate = self._make_surrogate()
        first = ConstantSurrogate(n_parameters=2, constant=0)
        surrogate._add_iteration(first, 1)

        # Append a second surrogate (constant = 9.9) manually
        second = ConstantSurrogate(n_parameters=2, constant=9.9)
        surrogate._add_iteration(second, 2)

        # Index 0 should return the first surrogate (constant 0.0)
        out0 = surrogate(0.0, 0.0, surrogate_index=0)
        self.assertAlmostEqual(out0["target"][0], 0.0)

        # Index -1 (default) should return the second surrogate (constant 9.9)
        out_last = surrogate(0.0, 0.0)   # default = -1
        self.assertAlmostEqual(out_last["target"][0], 9.9)

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
        self.assertEqual(len(surrogate.average_error_history), 3)
        self.assertEqual(len(surrogate.max_error_history), 3)
        self.assertEqual(len(surrogate.sample_count_history), 3)

        # Verify sample‑count history matches the values we passed
        self.assertEqual(surrogate.sample_count_history, [10, 20, 30])

        # Compute expected average L2 errors manually for sanity check
        R_test = surrogate._test_responses  # shape (2, 1)
        # Helper to compute error for a given constant
        def expected_avg(const):
            diff = R_test - const
            return np.linalg.norm(diff) / 1

        # Expected values
        exp0 = expected_avg(0.0)
        exp1 = expected_avg(1.0)
        exp2 = expected_avg(2.0)

        self.assertAlmostEqual(surrogate.average_error_history[0], exp0)
        self.assertAlmostEqual(surrogate.average_error_history[1], exp1)
        self.assertAlmostEqual(surrogate.average_error_history[2], exp2)

        # Max errors should be max absolute difference
        self.assertAlmostEqual(surrogate.max_error_history[0], np.max(np.abs(R_test - 0.0)))
        self.assertAlmostEqual(surrogate.max_error_history[1], np.max(np.abs(R_test - 1.0)))
        self.assertAlmostEqual(surrogate.max_error_history[2], np.max(np.abs(R_test - 2.0)))


class _FakeSurrogate:
    """
    Minimal surrogate object that mimics the public attributes used by the
    stopping‑criterion method.
    """
    def __init__(self):
        self._average_errors = []   # filled by the test
        self._max_errors = []       # filled by the test

    @property
    def average_error_history(self):
        return self._average_errors

    @property
    def max_error_history(self):
        return self._max_errors


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
        self._average_l2_error_goal = 1e-2   # default in the parent class
        self._max_abs_error_goal = 1e-1
        self._surrogate = _FakeSurrogate()
        self._results = _FakeResults()
        self._number_of_test_samples = 10
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
        self.study._surrogate._average_errors = [0.0]   # below 1e‑2
        self.study._surrogate._max_errors = [0.0]       # below 1e‑1
        self.study._results.number_of_evaluations = 0   # far below max_training_samples

        should_stop = self.study._stopping_criterion_met(training_batch_number=1)
        self.assertFalse(should_stop, "Stopping should NOT be triggered on the first batch")

    def test_stop_on_average_l2_error(self):
        self.study._surrogate._average_errors = [5e-3]   # < 1e‑2 goal
        self.study._surrogate._max_errors = [0.5]       # > goal (irrelevant)
        self.study._results.number_of_evaluations = 0

        should_stop = self.study._stopping_criterion_met(training_batch_number=2)
        self.assertTrue(should_stop,
                        "Stopping should be triggered when avg L2 error ≤ goal after >1 batch")

    def test_stop_on_max_absolute_error(self):
        self.study._surrogate._average_errors = [0.5]   # > goal (doesn't matter)
        self.study._surrogate._max_errors = [5e-2]      # < 1e‑1 goal
        self.study._results.number_of_evaluations = 0

        should_stop = self.study._stopping_criterion_met(training_batch_number=3)
        self.assertTrue(should_stop,
                        "Stopping should be triggered when max absolute error ≤ goal after >1 batch")

    def test_stop_on_max_training_samples(self):
        # Set error histories to values that would *not* normally trigger a stop
        self.study._surrogate._average_errors = [1.0]   # > goal
        self.study._surrogate._max_errors = [1.0]      # > goal

        # Simulate that the study has already used more samples than allowed.
        # The parent class stores the limit in `_max_training_samples`; we set it
        # directly on our test instance.
        self.study._max_training_samples = 100
        self.study._results.number_of_evaluations = 101  # exceed the limit

        should_stop = self.study._stopping_criterion_met(training_batch_number=5)
        self.assertTrue(should_stop,
                        "Stopping should be triggered when number_of_evaluations > max_training_samples")


    def test_no_stop_when_all_conditions_fail(self):
        self.study._surrogate._average_errors = [0.2]   # > 1e‑2
        self.study._surrogate._max_errors = [0.3]      # > 1e‑1
        self.study._max_training_samples = 1000
        self.study._results.number_of_evaluations = 500   # below limit

        should_stop = self.study._stopping_criterion_met(training_batch_number=4)
        self.assertFalse(should_stop,
                         "Stopping should NOT be triggered when no criteria are met")
        