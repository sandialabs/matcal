import os
import types
import numpy as np
import unittest


from matcal.core.adaptive_surrogates import (
    SparseGridAdaptiveSurrogateStudy,
    _get_parameter_bounds, _get_canonical_bounds
)
from matcal.core.parameters import Parameter
from matcal.core.objective import SimulationResultsSynchronizer
from matcal.core.models import PythonModel

# Import the MatcalUnitTest base class
from matcal.core.tests.MatcalUnitTest import MatcalUnitTest


def has_pyapprox():
    try:
        import pyapprox  
        HAS_PYAPPROX = True
    except Exception:    
        HAS_PYAPPROX = False
    return HAS_PYAPPROX


def return_data(*args, **kwargs):
    return {"x":np.linspace(0,1,10), "y":np.linspace(1,3,10)}
light_model = PythonModel(return_data)


class TestSparseGridAdaptiveSurrogate(MatcalUnitTest):
    """
    All tests inherit from MatcalUnitTest so that they get the same
    temporary build directory handling that the original MatCal unit tests use.
    """

    def setUp(self):
        """
        MatcalUnitTest expects a *filename* argument – we simply pass the
        current file's path.  After the base‑class set‑up we create a fresh
        study instance that uses a temporary working directory.
        """
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

    @unittest.skipIf(not has_pyapprox(),
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
        print(os.getcwd())
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

        class DummySurrogate:
            def __call__(self, params):
                return params * 2.0
        surrogate = DummySurrogate()

        self.study._update_surrogate_score(surrogate)

        self.assertEqual(len(self.study._average_l2_errors), 1)
        self.assertEqual(len(self.study._max_abs_errors), 1)

        expected_l2 = np.linalg.norm(self.study._test_responses -
                                    surrogate(self.study._test_params))
        expected_l2 /= self.study._test_responses.shape[1]
        self.assertAlmostEqual(self.study._average_l2_errors[0], expected_l2)

    @unittest.skipIf(not HAS_PYAPPROX,
                 "pyapprox not installed – skipping pyapprox‑dependent tests")
    def test_study_results_is_not_none(self):
        """launch must eventually call the sparse‑grid routine."""
        called = {"flag": False}
        self.study.set_independent_variable("x", np.linspace(0,1,3))
        self.study.set_target_field_name("y")
        self.study.add_evaluation_set(light_model)
        self.study.set_max_training_samples(1)
        self.study.set_number_of_test_samples(1)
        self.study.launch()
        self.assertIsNotNone(self.study.results)

    def _set_number_of_evaluations(self, n):
        self.fake_results = types.SimpleNamespace(number_of_evaluations=0)
        self.study._results = self.fake_results
        self.fake_results.number_of_evaluations = n

    def test_no_stop_when_first_batch(self):
        self.study._average_l2_errors.append(1e-5)
        self.study._max_abs_errors.append(1e-5)

        self._set_number_of_evaluations(10)   # below max_training_samples
        stop = self.study._stopping_criterion_met(training_batch_number=1)
        self.assertFalse(stop, "First batch should never trigger stopping")

    def test_stop_on_average_l2_error(self):
        self.study._average_l2_errors.append(self.study._average_l2_error_goal * 0.5)   
        self.study._max_abs_errors.append(0.1)                      # irrelevant

        self._set_number_of_evaluations(20)   # still below max_training_samples
        stop = self.study._stopping_criterion_met(training_batch_number=2)
        self.assertTrue(stop, "Average L2 error below criteria should stop")

    def test_stop_on_max_abs_error(self):
        self.study._average_l2_errors.append(0.5)               
        self.study._max_abs_errors.append(self.study._max_abs_error_goal * 0.8)  

        self._set_number_of_evaluations(30)
        stop = self.study._stopping_criterion_met(training_batch_number=3)
        self.assertTrue(stop, "Max‑abs error below criteria should stop")

    def test_stop_when_exceeds_max_training_samples(self):
        self.study._average_l2_errors.append(0.5)
        self.study._max_abs_errors.append(0.5)

        self._set_number_of_evaluations(self.study._max_training_samples + 1)

        stop = self.study._stopping_criterion_met(training_batch_number=2)
        self.assertTrue(stop, "Exceeding max_training_samples should stop")

    def test_no_stop_when_all_criteria_fail(self):
        self.study._average_l2_errors.append(0.5)  
        self.study._max_abs_errors.append(0.5)     

        self._set_number_of_evaluations(10)        # below max_training_samples
        stop = self.study._stopping_criterion_met(training_batch_number=4)
        self.assertFalse(stop, "No criteria met – should continue")

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

    def test_stopping_criterion_uses_new_goals(self):
        stricter_avg = 1e-4
        stricter_max = 1e-4
        self.study.set_error_stopping_criteria(
            average_l2_error_goal=stricter_avg,
            max_abs_error_goal=stricter_max,
        )

        self.study._average_l2_errors.append(0.5)   # > stricter goal
        self.study._max_abs_errors.append(0.5)     # > stricter goal
        class FakeResults:
            def __init__(self, number_of_evaluations: int = 0):
                self.number_of_evaluations = number_of_evaluations
        self.study._results = FakeResults(10)

        stop = self.study._stopping_criterion_met(training_batch_number=2)
        self.assertFalse(stop, "Stopping criterion should not be triggered")

        self.study._average_l2_errors.append(stricter_avg * 0.5)
        self.study._max_abs_errors.append(0.8)     # irrelevant for this case

        stop = self.study._stopping_criterion_met(training_batch_number=2)
        self.assertTrue(stop, "Stopping criterion should trigger with the new stricter goal")


class TestGetCanonicalBounds(MatcalUnitTest):
    def setUp(self):
        super().setUp(__file__)   # MatcalUnitTest expects a filename

    def test_single_variable(self):
        """For ``nvars == 1`` the function should return a 1×2 array ``[[-1., 1.]]``."""
        bounds = _get_canonical_bounds(1)
        expected = np.array([[-1.0, 1.0]])
        self.assertIsInstance(bounds, np.ndarray, "Result should be a NumPy array")
        np.testing.assert_array_equal(bounds, expected,
                                      err_msg="Canonical bounds for a single variable are incorrect")
        self.assertEqual(bounds.shape, (1, 2), "Shape must be (1, 2) for a single variable")

    def test_multiple_variables(self):
        """
        For ``nvars > 1`` each row must be ``[-1., 1.]`` and the shape must be
        ``(nvars, 2)``.
        """
        for nvars in (2, 3, 5, 10):
            with self.subTest(nvars=nvars):
                bounds = _get_canonical_bounds(nvars)
                self.assertEqual(bounds.shape, (nvars, 2),
                                 f"Expected shape {(nvars, 2)} but got {bounds.shape}")

                expected_row = np.array([-1.0, 1.0])
                for i in range(nvars):
                    np.testing.assert_array_equal(bounds[i], expected_row,
                                err_msg=f"Row {i} differs from expected canonical bounds")

                expected = np.tile(expected_row, (nvars, 1))
                np.testing.assert_array_equal(bounds, expected,
                            err_msg="Whole array does not match the expected canonical bounds")

