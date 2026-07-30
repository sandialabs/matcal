import os
import types
import numpy as np
import unittest
from unittest.mock import patch

from matcal.core.adaptive_surrogates import (
    _assign_points_to_nearest_seed,
    _calculate_native_cv_error,
    _create_ghost_points,
    _evaluate_kfold_cv_splits,
    _filter_points_within_bounds,
    _find_matching_row_index,
    _finite_vertex_indices,
    _get_parameter_bounds, 
    _make_bounded_nd_grid,
    _make_group_kfold_splits,
    _make_standard_kfold_splits,
    _normalize_candidate_array,
    _package_unique_bounded_points,
    _perform_kfold_cv,
    _perform_leave_one_out_cv,
    _random_subset_rows,
    _reduce_voronoi_candidates,
    _remove_duplicate_rows_against_existing,
    _remove_invalid_rows,
    _select_farthest_point,
    _validate_surrogate_storage_options,
    KFoldCrossValidation,
    LeaveOneOutCrossValidation,
    SparseGridAdaptiveSurrogateStudy,
    SparseGridAdaptiveSurrogate,
    VoronoiAdaptiveSurrogateStudy,
    VoronoiTessellation,    
)

from matcal.core.data import convert_dictionary_to_data
from matcal.core.objective import SimulationResultsSynchronizer
from matcal.core.models import PythonModel
from matcal.core.parameters import Parameter
from matcal.core.parameter_studies import HaltonStudy
from matcal.core.qoi_extractor import UserDefinedExtractor, InterpolatingExtractor
from matcal.core.serializer_wrapper import matcal_load
from matcal.core.study_base import StudyResults
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

    def test_score_is_defined_for_multiple_scalar_values(self):
        surrogate = self._make_surrogate()
        surrogate._add_iteration(
            ConstantSurrogate(n_parameters=2, constant=0.0),
            nsamples=5,
        )

        self.assertAlmostEqual(surrogate.score(), -9.0)

    def test_score_returns_nan_for_single_scalar_value(self):
        surrogate = SparseGridAdaptiveSurrogate(
            target_field_name="target",
            indep_variable_name="independent",
            indep_variable_values=np.array([0.0]),
            test_params=np.array([[0.0, 0.0]]),
            test_responses=np.array([[1.0]]),
            param_names=["p1", "p2"],
            bounds=np.array([[0.0, 1.0], [0.0, 1.0]]),
            storage_best_n_surrogates=None,
            storage_every_n_batches=1,
        )

        surrogate._add_iteration(
            ConstantSurrogate(n_parameters=2, constant=0.0),
            nsamples=5,
        )

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
        
    def _make_known_error_surrogate(self):
        """
        Build a multi-QoI sparse-grid adaptive surrogate with known test errors.

        Test responses are all zero, so the fixed surrogate prediction is also
        the signed error.

        Per-sample max absolute errors are:
            sample 0: 3.0
            sample 1: 5.0
            sample 2: 0.5
            sample 3: 2.0
            sample 4: 7.0
            sample 5: 4.0
            sample 6: 6.0

        Therefore worst-to-best by max_error is:
            [4, 6, 1, 5, 0, 3, 2]
        """
        param_names = ["p1", "p2"]

        test_params = np.column_stack((
            np.linspace(0.0, 1.0, 7),
            np.linspace(1.0, 0.0, 7),
        ))

        test_responses = np.zeros((7, 3))

        surrogate = SparseGridAdaptiveSurrogate(
            target_field_name="target",
            indep_variable_name="independent",
            indep_variable_values=np.array([0.0, 0.5, 1.0]),
            test_params=test_params,
            test_responses=test_responses,
            param_names=param_names,
            bounds=np.array([[0.0, 1.0], [0.0, 1.0]]),
            storage_best_n_surrogates=None,
            storage_every_n_batches=1,
        )

        fixed_prediction = np.array([
            [1.0, 2.0, 3.0],
            [-5.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
            [2.0, 2.0, 1.0],
            [7.0, 0.0, 0.0],
            [4.0, 1.0, 1.0],
            [6.0, 0.0, 0.0],
        ])

        surrogate._add_iteration(
            FixedResponseSurrogate(fixed_prediction),
            nsamples=10,
        )

        return surrogate, fixed_prediction

    def test_plot_worst_N_splits_samples_across_requested_number_of_figures(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        surrogate, _ = self._make_known_error_surrogate()

        figures, axes_groups, worst_indices = surrogate.plot_worst_N(
            N=7,
            n_figures=3,
            surrogate_index="best",
            metric="max_error",
            error_type="signed",
            independent_variable_units="s",
            target_field_units="K",
        )

        np.testing.assert_array_equal(
            worst_indices,
            np.array([4, 6, 1, 5, 0, 3, 2]),
        )

        self.assertEqual(len(figures), 3)
        self.assertEqual(len(axes_groups), 3)

        for axes in axes_groups:
            self.assertEqual(axes.shape, (1, 2))

        # np.array_split splits 7 samples over 3 figures as 3, 2, 2.
        expected_group_sizes = [3, 2, 2]

        for axes, group_size in zip(axes_groups, expected_group_sizes):
            response_axes = axes[0, 0]
            error_axes = axes[0, 1]

            # Each sample contributes one test curve and one surrogate curve.
            self.assertEqual(len(response_axes.lines), 2 * group_size)

            # Each sample contributes one error curve. Signed error also adds a
            # zero-reference line.
            self.assertEqual(len(error_axes.lines), group_size + 1)

        self.assertIn("Worst samples 1-3", axes_groups[0][0, 0].get_title())
        self.assertIn("Worst samples 4-5", axes_groups[1][0, 0].get_title())
        self.assertIn("Worst samples 6-7", axes_groups[2][0, 0].get_title())

        for fig in figures:
            plt.close(fig)

    def test_plot_worst_N_uses_common_axis_limits_across_figures(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        surrogate, _ = self._make_known_error_surrogate()

        figures, axes_groups, _ = surrogate.plot_worst_N(
            N=7,
            n_figures=3,
            surrogate_index="best",
            metric="max_error",
            error_type="signed",
            independent_variable_units="s",
            target_field_units="K",
        )

        response_xlims = [axes[0, 0].get_xlim() for axes in axes_groups]
        response_ylims = [axes[0, 0].get_ylim() for axes in axes_groups]
        error_xlims = [axes[0, 1].get_xlim() for axes in axes_groups]
        error_ylims = [axes[0, 1].get_ylim() for axes in axes_groups]

        for xlim in response_xlims[1:]:
            self.assertEqual(xlim, response_xlims[0])
        for ylim in response_ylims[1:]:
            self.assertEqual(ylim, response_ylims[0])
        for xlim in error_xlims[1:]:
            self.assertEqual(xlim, error_xlims[0])
        for ylim in error_ylims[1:]:
            self.assertEqual(ylim, error_ylims[0])

        for fig in figures:
            plt.close(fig)

    def test_plot_worst_N_clips_number_of_figures_to_number_of_samples(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        surrogate, _ = self._make_known_error_surrogate()

        figures, axes_groups, worst_indices = surrogate.plot_worst_N(
            N=3,
            n_figures=10,
            surrogate_index="best",
            metric="max_error",
            error_type="signed",
        )

        np.testing.assert_array_equal(
            worst_indices,
            np.array([4, 6, 1]),
        )

        # Only three samples are plotted, so only three figures are created.
        self.assertEqual(len(figures), 3)
        self.assertEqual(len(axes_groups), 3)

        for axes in axes_groups:
            response_axes = axes[0, 0]
            error_axes = axes[0, 1]

            # One sample per figure.
            self.assertEqual(len(response_axes.lines), 2)
            self.assertEqual(len(error_axes.lines), 2)

        for fig in figures:
            plt.close(fig)

    def test_plot_worst_N_clips_N_larger_than_number_of_test_samples(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        surrogate, _ = self._make_known_error_surrogate()

        figures, axes_groups, worst_indices = surrogate.plot_worst_N(
            N=100,
            n_figures=2,
            surrogate_index="best",
            metric="max_error",
            error_type="signed",
        )

        np.testing.assert_array_equal(
            worst_indices,
            np.array([4, 6, 1, 5, 0, 3, 2]),
        )

        # Seven available samples split over two figures as 4 and 3.
        self.assertEqual(len(figures), 2)
        self.assertEqual(len(axes_groups), 2)

        first_response_axes = axes_groups[0][0, 0]
        first_error_axes = axes_groups[0][0, 1]
        second_response_axes = axes_groups[1][0, 0]
        second_error_axes = axes_groups[1][0, 1]

        self.assertEqual(len(first_response_axes.lines), 8)
        self.assertEqual(len(first_error_axes.lines), 5)
        self.assertEqual(len(second_response_axes.lines), 6)
        self.assertEqual(len(second_error_axes.lines), 4)

        for fig in figures:
            plt.close(fig)

    def test_plot_worst_N_absolute_error_does_not_add_zero_reference_line(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        surrogate, _ = self._make_known_error_surrogate()

        figures, axes_groups, worst_indices = surrogate.plot_worst_N(
            N=5,
            n_figures=1,
            surrogate_index="best",
            metric="max_error",
            error_type="absolute",
            independent_variable_units="s",
            target_field_units="K",
        )

        np.testing.assert_array_equal(
            worst_indices,
            np.array([4, 6, 1, 5, 0]),
        )

        response_axes = axes_groups[0][0, 0]
        error_axes = axes_groups[0][0, 1]

        self.assertEqual(len(response_axes.lines), 10)

        # Absolute-error plot should have only one error curve per sample.
        # There should be no zero-reference line.
        self.assertEqual(len(error_axes.lines), 5)
        self.assertEqual(error_axes.get_ylabel(), "target absolute error (K)")

        plt.close(figures[0])

    def test_plot_worst_N_rejects_invalid_number_of_figures(self):
        surrogate, _ = self._make_known_error_surrogate()

        with self.assertRaises(ValueError):
            surrogate.plot_worst_N(N=5, n_figures=0)

        with self.assertRaises(TypeError):
            surrogate.plot_worst_N(N=5, n_figures="bad")

    def test_plot_worst_N_rejects_invalid_inputs(self):
        surrogate, _ = self._make_known_error_surrogate()

        with self.assertRaises(ValueError):
            surrogate.plot_worst_N(N=0)

        with self.assertRaises(TypeError):
            surrogate.plot_worst_N(N="bad")

        with self.assertRaises(ValueError):
            surrogate.plot_worst_N(metric="bad_metric")

        with self.assertRaises(ValueError):
            surrogate.plot_worst_N(error_type="bad_error_type")

    def test_plot_worst_N_no_longer_accepts_figure_or_axes_arguments(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        surrogate, _ = self._make_known_error_surrogate()

        fig, axes = plt.subplots(1, 2)

        with self.assertRaises(TypeError):
            surrogate.plot_worst_N(
                N=2,
                figure=fig,
            )

        with self.assertRaises(TypeError):
            surrogate.plot_worst_N(
                N=2,
                axes=axes,
            )

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

class TestGlobalHelperFunctions(MatcalUnitTest):

    def setUp(self):
        super().setUp(__file__)

    def test_validate_surrogate_storage_options_normalizes_score_alias(self):
        best_n, every_n, metric = _validate_surrogate_storage_options(
            best_n_surrogates=None,
            save_every_n_batches=2,
            score_metric="score",
        )

        self.assertIsNone(best_n)
        self.assertEqual(every_n, 2)
        self.assertEqual(metric, "r2")
    def test_validate_surrogate_storage_options_rejects_disabled_retention(self):
        with self.assertRaises(ValueError):
            _validate_surrogate_storage_options(
                best_n_surrogates=None,
                save_every_n_batches=None,
            )        

    def test_package_unique_bounded_points_uniques_and_filters_bounds(self):
        bounds = np.array([
            [0.0, 1.0],
            [-1.0, 1.0],
        ])

        points = np.array([
            [0.5, 0.0],
            [0.5, 0.0],
            [1.2, 0.0],
            [0.25, -0.5],
            [np.nan, 0.0],
        ])

        packaged = _package_unique_bounded_points(
            points,
            bounds,
            n_parameters=2,
        )

        expected = np.array([
            [0.25, -0.5],
            [0.5, 0.0],
        ])

        self.assert_close_arrays(packaged, expected)

class TestVoronoiPureFunctions(MatcalUnitTest):

    def setUp(self):
        super().setUp(__file__)

    def test_make_bounded_nd_grid_returns_cartesian_grid(self):
        bounds = np.array([
            [0.0, 1.0],
            [10.0, 20.0],
        ])

        grid = _make_bounded_nd_grid(bounds, 2)

        expected_points = np.array([
            [0.0, 10.0],
            [1.0, 10.0],
            [0.0, 20.0],
            [1.0, 20.0],
        ])

        self.assertEqual(grid.shape, (4, 2))

        for expected_point in expected_points:
            self.assertTrue(
                np.any(np.all(np.isclose(grid, expected_point), axis=1)),
                msg=f"Expected point {expected_point} was not in grid {grid}.",
            )

    def test_make_bounded_nd_grid_rejects_bad_bounds_shape(self):
        bad_bounds = np.array([0.0, 1.0])

        with self.assertRaises(ValueError):
            _make_bounded_nd_grid(bad_bounds, 2)

    def test_normalize_candidate_array_handles_empty_input(self):
        normalized = _normalize_candidate_array([], 3)

        self.assertEqual(normalized.shape, (0, 3))

    def test_normalize_candidate_array_handles_single_point(self):
        normalized = _normalize_candidate_array([1.0, 2.0, 3.0], 3)

        expected = np.array([[1.0, 2.0, 3.0]])
        self.assert_close_arrays(normalized, expected)

    def test_normalize_candidate_array_rejects_wrong_dimension(self):
        with self.assertRaises(ValueError):
            _normalize_candidate_array([[1.0], [2.0]], 2)

    def test_filter_points_within_bounds_removes_nan_inf_and_out_of_bounds(self):
        bounds = np.array([
            [0.0, 1.0],
            [-1.0, 1.0],
        ])
        points = np.array([
            [0.5, 0.0],
            [1.0, 1.0],
            [1.1, 0.0],
            [0.5, np.nan],
            [0.5, np.inf],
            [0.0, -1.0],
        ])

        filtered = _filter_points_within_bounds(points, bounds, 2)

        expected = np.array([
            [0.5, 0.0],
            [1.0, 1.0],
            [0.0, -1.0],
        ])

        self.assert_close_arrays(filtered, expected)

    def test_remove_duplicate_rows_against_existing_preserves_order(self):
        existing_points = np.array([
            [0.0, 0.0],
            [1.0, 1.0],
        ])
        candidate_points = np.array([
            [1.0, 1.0 + 5.0e-11],  # duplicate of existing point
            [0.5, 0.5],
            [0.5, 0.5 + 5.0e-11],  # duplicate of prior kept candidate
            [2.0, 2.0],
        ])

        filtered = _remove_duplicate_rows_against_existing(
            candidate_points,
            existing_points,
            n_parameters=2,
            atol=1.0e-10,
        )

        expected = np.array([
            [0.5, 0.5],
            [2.0, 2.0],
        ])

        self.assert_close_arrays(filtered, expected)

    def test_random_subset_rows_is_reproducible(self):
        values = np.array([
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
            [4.0, 4.0],
        ])

        rng_a = np.random.default_rng(123)
        rng_b = np.random.default_rng(123)

        subset_a = _random_subset_rows(values, 3, rng_a, 2)
        subset_b = _random_subset_rows(values, 3, rng_b, 2)

        self.assert_close_arrays(subset_a, subset_b)
        self.assertEqual(subset_a.shape, (3, 2))

        for row in subset_a:
            self.assertTrue(np.any(np.all(np.isclose(values, row), axis=1)))

    def test_find_matching_row_index_returns_first_matching_row(self):
        rows = np.array([
            [0.0, 0.0],
            [1.0, 1.0],
            [1.0, 1.0 + 5.0e-11],
        ])

        index = _find_matching_row_index(rows, [1.0, 1.0], atol=1.0e-10)

        self.assertEqual(index, 1)

    def test_find_matching_row_index_returns_none_when_no_match(self):
        rows = np.array([
            [0.0, 0.0],
            [1.0, 1.0],
        ])

        index = _find_matching_row_index(rows, [2.0, 2.0])

        self.assertIsNone(index)

    def test_assign_points_to_nearest_seed(self):
        query_points = np.array([
            [0.1, 0.0],
            [0.9, 0.0],
            [2.2, 0.0],
        ])
        seed_points = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
        ])

        owner_indices = _assign_points_to_nearest_seed(query_points, seed_points)

        expected = np.array([0, 1, 2])
        self.assert_close_arrays(owner_indices, expected)

    def test_select_farthest_point(self):
        candidate_points = np.array([
            [0.1, 0.0],
            [0.5, 0.0],
            [2.0, 0.0],
        ])
        seed_location = np.array([0.0, 0.0])

        farthest = _select_farthest_point(candidate_points, seed_location)

        expected = np.array([2.0, 0.0])
        self.assert_close_arrays(farthest, expected)

    def test_select_farthest_point_returns_none_for_empty_candidates(self):
        farthest = _select_farthest_point(np.empty((0, 2)), [0.0, 0.0])

        self.assertIsNone(farthest)

    def test_create_ghost_points_for_non_centered_domain_are_outside_domain(self):
        bounds = np.array([
            [2.0, 4.0],
            [10.0, 20.0],
        ])
        boundary_points = _make_bounded_nd_grid(bounds, 2)

        ghost_points = _create_ghost_points(boundary_points, n_dimensions=2)

        inside_mask = (
            (ghost_points >= bounds[:, 0])
            & (ghost_points <= bounds[:, 1])
        ).all(axis=1)

        self.assertFalse(np.any(inside_mask))

    def test_finite_vertex_indices_keeps_zero_and_removes_negative_markers(self):
        region = [-2, -1, 0, 3, 7]

        finite_indices = _finite_vertex_indices(region)

        self.assertEqual(finite_indices, [0, 3, 7])

    def test_reduce_voronoi_candidates_preserves_order_in_one_at_a_time_mode(self):
        candidates = np.array([
            [0.0, -1.0],
            [0.25, -0.5],
            [0.5, 0.0],
        ])

        reduced = _reduce_voronoi_candidates(
            candidates,
            n_parameters=2,
            batch_size=1,
            thin=2,
            random_selection=None,
            random_generator=np.random.default_rng(123),
        )

        self.assert_close_arrays(reduced, candidates)

    def test_reduce_voronoi_candidates_uses_random_selection_before_batch_limit(self):
        candidates = np.array([
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
            [4.0, 4.0],
            [5.0, 5.0],
        ])

        rng = np.random.default_rng(123)

        reduced = _reduce_voronoi_candidates(
            candidates,
            n_parameters=2,
            batch_size=2,
            thin=None,
            random_selection=4,
            random_generator=rng,
        )

        self.assertEqual(reduced.shape, (2, 2))

        for row in reduced:
            self.assertTrue(
                np.any(np.all(np.isclose(candidates, row), axis=1))
            )


class TestVoronoiPhysicalCrossValidationHelpers(MatcalUnitTest):

    def setUp(self):
        super().setUp(__file__)

    def test_calculate_native_cv_error_is_zero_for_exact_fake_surrogate(self):

        class ExactFakeSurrogate:

            def __call__(self, X, batch_evaluate=False):
                X = np.asarray(X, dtype=float)
                response = np.column_stack([
                    X[:, 0] + X[:, 1],
                    X[:, 0] - X[:, 1],
                ])
                return {"response": response}

        X_test = np.array([
            [1.0, 2.0],
            [3.0, 4.0],
        ])

        y_test = [
            {
                "time": np.array([0.0, 1.0]),
                "response": np.array([3.0, -1.0]),
            },
            {
                "time": np.array([0.0, 1.0]),
                "response": np.array([7.0, -1.0]),
            },
        ]

        error = _calculate_native_cv_error(
            ExactFakeSurrogate(),
            X_test,
            y_test,
            target_field="response",
            interpolation_field="time",
            interpolation_values=np.array([0.0, 1.0]),
            metric="rmse",
            scale=1.0,
        )

        self.assertAlmostEqual(error, 0.0)

    def test_calculate_native_cv_error_sum_abs(self):

        class BiasedFakeSurrogate:

            def __call__(self, X, batch_evaluate=False):
                X = np.asarray(X, dtype=float)
                response = np.column_stack([
                    X[:, 0] + X[:, 1] + 1.0,
                    X[:, 0] - X[:, 1] - 2.0,
                ])
                return {"response": response}

        X_test = np.array([
            [1.0, 2.0],
        ])

        y_test = [
            {
                "time": np.array([0.0, 1.0]),
                "response": np.array([3.0, -1.0]),
            },
        ]

        error = _calculate_native_cv_error(
            BiasedFakeSurrogate(),
            X_test,
            y_test,
            target_field="response",
            interpolation_field="time",
            interpolation_values=np.array([0.0, 1.0]),
            metric="sum_abs",
            scale=1.0,
        )

        # Residual = true - predicted = [-1, 2], sum(abs(residual)) = 3.
        self.assertAlmostEqual(error, 3.0)


class TestLeaveOneOutCrossValidation(MatcalUnitTest):

    def setUp(self):
        super().setUp(__file__)

    def _make_loocv(self):
        return LeaveOneOutCrossValidation(
            scale=1.0,
            metric="sum_abs",
            interpolation_field="time",
            interpolation_values=np.array([0.0, 1.0]),
            target_field="response",
            par_names=["x", "y"],
            surrogate_options={"surrogate_type": "fake"},
        )

    def _make_training_arrays(self):
        X = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ])

        y = [
            {
                "time": np.array([0.0, 1.0]),
                "response": np.array([0.0, 1.0]),
            },
            {
                "time": np.array([0.0, 1.0]),
                "response": np.array([1.0, 2.0]),
            },
            {
                "time": np.array([0.0, 1.0]),
                "response": np.array([2.0, 3.0]),
            },
        ]

        return X, y

    def _assert_parameter_history_matches_samples(self, study_results, samples):
        """
        Assert that a StudyResults parameter history matches a sample matrix.

        StudyResults stores parameter history in parameter-major form:

        ``parameter_history["x"]``, ``parameter_history["y"]``, ...

        rather than evaluation-major form:

        ``parameter_history["eval_0"]["x"]``.
        """
        samples = np.asarray(samples, dtype=float)

        x_values = np.asarray(study_results.parameter_history["x"], dtype=float)
        y_values = np.asarray(study_results.parameter_history["y"], dtype=float)

        self.assert_close_arrays(x_values, samples[:, 0])
        self.assert_close_arrays(y_values, samples[:, 1])

    def test_extract_loo_info_removes_selected_sample_from_training_set(self):
        loocv = self._make_loocv()
        X, y = self._make_training_arrays()

        train_res, test_res, X_test, y_test = loocv.extract_loo_info(1, X, y)

        expected_X_train = np.delete(X, 1, axis=0)
        expected_X_test = X[[1], :]

        self.assertIsInstance(train_res, StudyResults)
        self.assertIsInstance(test_res, StudyResults)

        self.assert_close_arrays(X_test, expected_X_test)

        self.assertEqual(len(y_test), 1)
        self.assert_close_arrays(
            y_test[0]["response"],
            y[1]["response"],
        )

        self._assert_parameter_history_matches_samples(
            train_res,
            expected_X_train,
        )
        self._assert_parameter_history_matches_samples(
            test_res,
            expected_X_test,
        )

    def test_evaluate_loo_sample_uses_native_cv_error_function(self):
        loocv = self._make_loocv()
        X, y = self._make_training_arrays()

        fake_surrogate = object()

        with patch(
            "matcal.core.adaptive_surrogates._fit_surrogate_model",
            return_value=fake_surrogate,
        ) as fit_mock:
            with patch(
                "matcal.core.adaptive_surrogates._calculate_native_cv_error",
                return_value=12.5,
            ) as error_mock:
                error, omitted_index = loocv.evaluate_loo_sample(X, y, 1)

        self.assertEqual(error, 12.5)
        self.assertEqual(omitted_index, 1)

        fit_mock.assert_called_once()
        error_mock.assert_called_once()

        error_args = error_mock.call_args[0]
        self.assertIs(error_args[0], fake_surrogate)
        self.assert_close_arrays(error_args[1], X[[1], :])
        self.assertEqual(error_args[2], [y[1]])
        self.assertEqual(error_args[3], "response")
        self.assertEqual(error_args[4], "time")
        self.assert_close_arrays(error_args[5], np.array([0.0, 1.0]))
        self.assertEqual(error_args[6], "sum_abs")
        self.assertEqual(error_args[7], 1.0)

    def test_perform_loocv_returns_error_and_original_sample_index(self):
        loocv = self._make_loocv()
        X, y = self._make_training_arrays()

        def fake_evaluate_loo_sample(_, __, index):
            return float(index + 10), index

        loocv.evaluate_loo_sample = fake_evaluate_loo_sample

        results = _perform_leave_one_out_cv(
            loocv,
            X,
            y,
            indices=np.array([2, 0]),
        )
        self.assertEqual(results[0], (12.0, 2))
        self.assertEqual(results[1], (10.0, 0))


class TestKFoldCrossValidation(MatcalUnitTest):

    def setUp(self):
        super().setUp(__file__)

    def _make_kfold_cv(self, nsplits=2, group_kfold=False, random_seed=123):
        return KFoldCrossValidation(
            nsplits=nsplits,
            group_kfold=group_kfold,
            interpolation_field="time",
            interpolation_values=np.array([0.0, 1.0]),
            scale=1.0,
            metric="rmse",
            target_field="response",
            param_names=["x", "y"],
            surrogate_options={"surrogate_type": "fake"},
            random_seed=random_seed,
        )

    def _make_training_arrays(self):
        X = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ])

        y = [
            {
                "time": np.array([0.0, 1.0]),
                "response": np.array([0.0, 1.0]),
            },
            {
                "time": np.array([0.0, 1.0]),
                "response": np.array([1.0, 2.0]),
            },
            {
                "time": np.array([0.0, 1.0]),
                "response": np.array([2.0, 3.0]),
            },
            {
                "time": np.array([0.0, 1.0]),
                "response": np.array([3.0, 4.0]),
            },
        ]

        return X, y

    def _assert_parameter_history_matches_samples(self, study_results, samples):
        """
        Assert that a StudyResults parameter history matches a sample matrix.

        StudyResults stores parameter history in parameter-major form:

        ``parameter_history["x"]``, ``parameter_history["y"]``, ...

        rather than evaluation-major form:

        ``parameter_history["eval_0"]["x"]``.
        """
        samples = np.asarray(samples, dtype=float)

        x_values = np.asarray(study_results.parameter_history["x"], dtype=float)
        y_values = np.asarray(study_results.parameter_history["y"], dtype=float)

        self.assert_close_arrays(x_values, samples[:, 0])
        self.assert_close_arrays(y_values, samples[:, 1])

    def test_make_standard_kfold_splits_is_reproducible(self):
        X, _ = self._make_training_arrays()

        kfcv_a = self._make_kfold_cv(
            nsplits=2,
            group_kfold=False,
            random_seed=42,
        )
        kfcv_b = self._make_kfold_cv(
            nsplits=2,
            group_kfold=False,
            random_seed=42,
        )

        splits_a = list(
            _make_standard_kfold_splits(
                X,
                kfcv_a.nsplits,
                kfcv_a.random_seed,
            )
        )
        splits_b = list(
            _make_standard_kfold_splits(
                X,
                kfcv_b.nsplits,
                kfcv_b.random_seed,
            )
        )

        self.assertEqual(len(splits_a), 2)
        self.assertEqual(len(splits_b), 2)

        for (train_a, test_a), (train_b, test_b) in zip(splits_a, splits_b):
            self.assert_close_arrays(train_a, train_b)
            self.assert_close_arrays(test_a, test_b)

    def test_make_group_kfold_splits_requires_groups(self):
        X, y = self._make_training_arrays()
        kfcv = self._make_kfold_cv(nsplits=2, group_kfold=True)

        with self.assertRaises(RuntimeError):
            list(
                _make_group_kfold_splits(
                    X,
                    y,
                    groups=None,
                    nsplits=kfcv.nsplits,
                )
            )

    def test_make_group_kfold_splits_uses_group_labels(self):
        X, y = self._make_training_arrays()
        kfcv = self._make_kfold_cv(nsplits=2, group_kfold=True)

        groups = np.array([0, 0, 1, 1])

        splits = list(
            _make_group_kfold_splits(
                X,
                y,
                groups,
                kfcv.nsplits,
            )
        )
        self.assertEqual(len(splits), 2)

        for _, test_index in splits:
            test_groups = np.unique(groups[test_index])
            self.assertEqual(test_groups.size, 1)

    def test_extract_fold_info_partitions_training_and_test_data(self):
        X, y = self._make_training_arrays()
        kfcv = self._make_kfold_cv(nsplits=2)

        train_index = np.array([0, 2])
        test_index = np.array([1, 3])

        train_res, test_res, X_test, y_test = kfcv.extract_fold_info(
            train_index,
            test_index,
            X,
            y,
        )

        expected_X_train = X[train_index]
        expected_X_test = X[test_index]
        expected_y_test = [y[1], y[3]]

        self.assertIsInstance(train_res, StudyResults)
        self.assertIsInstance(test_res, StudyResults)

        self.assert_close_arrays(X_test, expected_X_test)

        self.assertEqual(len(y_test), len(expected_y_test))
        for actual, expected in zip(y_test, expected_y_test):
            self.assert_close_arrays(
                actual["response"],
                expected["response"],
            )

        self._assert_parameter_history_matches_samples(
            train_res,
            expected_X_train,
        )
        self._assert_parameter_history_matches_samples(
            test_res,
            expected_X_test,
        )

    def test_evaluate_fold_uses_native_cv_error_function(self):
        X, y = self._make_training_arrays()
        kfcv = self._make_kfold_cv(nsplits=2)

        train_index = np.array([0, 2])
        test_index = np.array([1, 3])
        fake_surrogate = object()

        with patch(
            "matcal.core.adaptive_surrogates._fit_surrogate_model",
            return_value=fake_surrogate,
        ) as fit_mock:
            with patch(
                "matcal.core.adaptive_surrogates._calculate_native_cv_error",
                return_value=4.25,
            ) as error_mock:
                error, returned_test_index = kfcv.evaluate_fold(
                    train_index,
                    test_index,
                    X,
                    y,
                    kfold_count=0,
                )

        self.assertEqual(error, 4.25)
        self.assert_close_arrays(returned_test_index, test_index)

        fit_mock.assert_called_once()
        error_mock.assert_called_once()

        error_args = error_mock.call_args[0]
        self.assertIs(error_args[0], fake_surrogate)
        self.assert_close_arrays(error_args[1], X[test_index])
        self.assertEqual(error_args[2], [y[1], y[3]])
        self.assertEqual(error_args[3], "response")
        self.assertEqual(error_args[4], "time")
        self.assert_close_arrays(error_args[5], np.array([0.0, 1.0]))
        self.assertEqual(error_args[6], "rmse")
        self.assertEqual(error_args[7], 1.0)

    def test_evaluate_cv_splits_returns_fold_dictionary(self):
        X, y = self._make_training_arrays()
        kfcv = self._make_kfold_cv(nsplits=2)

        splits = [
            (np.array([0, 1]), np.array([2, 3])),
            (np.array([2, 3]), np.array([0, 1])),
        ]

        def fake_evaluate_fold(train_index, test_index, X_arg, y_arg, kfold_count):
            return float(kfold_count), test_index

        kfcv.evaluate_fold = fake_evaluate_fold

        results = _evaluate_kfold_cv_splits(kfcv, X, y, splits)
        self.assertEqual(set(results.keys()), {0, 1})
        self.assertEqual(results[0][0], 0.0)
        self.assertEqual(results[1][0], 1.0)
        self.assert_close_arrays(results[0][1], np.array([2, 3]))
        self.assert_close_arrays(results[1][1], np.array([0, 1]))

    def test_perform_kfold_cv_uses_global_split_and_evaluation_helpers(self):
        X, y = self._make_training_arrays()
        kfcv = self._make_kfold_cv(nsplits=2)

        def fake_evaluate_fold(train_index, test_index, X_arg, y_arg, kfold_count):
            return float(kfold_count), test_index

        kfcv.evaluate_fold = fake_evaluate_fold

        results = _perform_kfold_cv(kfcv, X, y)

        self.assertEqual(set(results.keys()), {0, 1})
        self.assertEqual(results[0][0], 0.0)
        self.assertEqual(results[1][0], 1.0)


class TestVoronoiTessellation(MatcalUnitTest):

    def setUp(self):
        super().setUp(__file__)

    def _make_points(self):
        return np.array([
            [0.25, 0.25],
            [0.75, 0.25],
            [0.25, 0.75],
            [0.75, 0.75],
        ])

    def _make_bounds(self):
        return np.array([
            [0.0, 1.0],
            [0.0, 1.0],
        ])

    def _make_tessellation(self, finite_only=False):
        return VoronoiTessellation(
            self._make_points(),
            self._make_bounds(),
            finite_only=finite_only,
        )
  
    def _expected_number_of_ghost_points(self, tessellation):
        """
        Match the ghost-point construction used by _create_ghost_points.

        _create_ghost_points returns:

        * one stretched ghost point for each boundary point; and
        * positive/negative axis ghost points for each dimension.
        """
        return tessellation.boundary_points.shape[0] + 2 * tessellation.ndim

    def test_module_level_remove_invalid_rows_removes_nonfinite_rows(self):
        values = np.array([
            [0.0, 0.0],
            [np.nan, 1.0],
            [1.0, np.inf],
            [2.0, 2.0],
        ])

        filtered = _remove_invalid_rows(values)

        expected = np.array([
            [0.0, 0.0],
            [2.0, 2.0],
        ])

        self.assert_close_arrays(filtered, expected)

    def test_build_creates_expected_boundary_points(self):
        tessellation = self._make_tessellation(finite_only=False)

        tessellation.build()

        expected_boundary_points = _make_bounded_nd_grid(self._make_bounds(), 2)

        self.assert_close_arrays(
            tessellation.boundary_points,
            expected_boundary_points,
        )

    def test_build_uses_native_and_ghost_points_in_voronoi_object(self):
        tessellation = self._make_tessellation(finite_only=False)

        tessellation.build()

        n_native_points = tessellation.points.shape[0]
        n_ghost_points = self._expected_number_of_ghost_points(tessellation)
        expected_total_points = n_native_points + n_ghost_points

        self.assertEqual(tessellation.vor.points.shape[0], expected_total_points)

    def test_build_marks_native_and_ghost_points_in_boolean_mask(self):
        tessellation = self._make_tessellation(finite_only=False)

        tessellation.build()

        n_native_points = tessellation.points.shape[0]
        n_ghost_points = self._expected_number_of_ghost_points(tessellation)

        self.assertEqual(len(tessellation._boo), n_native_points + n_ghost_points)

        # The tessellation is constructed with native points followed by local
        # ghost points, so the first block should be native and the second
        # block should be ghost points.
        self.assertEqual(
            tessellation._boo[:n_native_points],
            [False] * n_native_points,
        )
        self.assertEqual(
            tessellation._boo[n_native_points:],
            [True] * n_ghost_points,
        )

    def test_get_voronoi_region_returns_region_for_existing_seed(self):
        tessellation = self._make_tessellation(finite_only=False)
        tessellation.build()

        seed = tessellation.points[0]
        region = tessellation.get_voronoi_region(seed)

        expected_region = tessellation.vor.point_region[0]

        self.assertEqual(region, [[int(expected_region)]])

    def test_get_region_vertices_returns_finite_vertices_inside_bounds(self):
        tessellation = self._make_tessellation(finite_only=False)
        tessellation.build()

        seed = tessellation.points[0]
        region_index = tessellation.get_voronoi_region(seed)[0][0]

        vertices = tessellation.get_region_vertices(
            region_index,
            identify_outside_vertices=True,
        )

        self.assertIsNotNone(vertices)
        self.assertTrue(np.isfinite(vertices).all())
        self.assertTrue(np.all(vertices >= tessellation.bounds[:, 0]))
        self.assertTrue(np.all(vertices <= tessellation.bounds[:, 1]))

    def test_get_voronoi_vertices_skips_ghost_regions_and_returns_bounded_vertices(self):
        tessellation = self._make_tessellation(finite_only=False)
        tessellation.build()

        vertices = tessellation.get_voronoi_vertices(
            identify_outside_vertices=True,
        )

        self.assertIsNotNone(vertices)
        self.assertEqual(vertices.shape[1], 2)
        self.assertTrue(np.isfinite(vertices).all())
        self.assertTrue(np.all(vertices >= tessellation.bounds[:, 0]))
        self.assertTrue(np.all(vertices <= tessellation.bounds[:, 1]))

    def test_add_points_discards_invalid_rows_and_removes_duplicates(self):
        tessellation = self._make_tessellation(finite_only=False)
        tessellation.build()

        original_points = tessellation.points.copy()

        new_points = np.array([
            [0.5, 0.5],
            [0.5, 0.5],
            [np.nan, 0.0],
            [np.inf, 0.0],
        ])

        tessellation.add_points(new_points)

        self.assertTrue(np.isfinite(tessellation.points).all())
        self.assertEqual(
            tessellation.points.shape[0],
            original_points.shape[0] + 1,
        )
        self.assertTrue(
            np.any(np.all(np.isclose(tessellation.points, [0.5, 0.5]), axis=1))
        )

    def test_add_points_with_empty_array_leaves_points_unchanged(self):
        tessellation = self._make_tessellation(finite_only=False)
        tessellation.build()

        original_points = tessellation.points.copy()

        tessellation.add_points(np.empty((0, 2)))

        self.assert_close_arrays(tessellation.points, original_points)

    def test_add_points_rebuilds_cached_voronoi_state_without_persisting_ghost_points(self):
        tessellation = self._make_tessellation(finite_only=False)
        tessellation.build()

        tessellation.add_points(np.array([[0.5, 0.5]]))

        self.assertTrue(hasattr(tessellation, "vor"))
        self.assertTrue(hasattr(tessellation, "_boo"))
        self.assertTrue(hasattr(tessellation, "boundary_regions"))
        self.assertFalse(hasattr(tessellation, "_ghost_points"))
        self.assertFalse(hasattr(tessellation, "ghost_points"))

        n_native_points = tessellation.points.shape[0]
        n_ghost_points = self._expected_number_of_ghost_points(tessellation)

        self.assertEqual(len(tessellation._boo), n_native_points + n_ghost_points)
        self.assertEqual(
            tessellation.vor.points.shape[0],
            n_native_points + n_ghost_points,
        )
        self.assertEqual(
            len(tessellation.boundary_regions),
            tessellation.boundary_points.shape[0],
        )

    def test_add_points_rebuilds_boolean_mask_after_point_addition(self):
        tessellation = self._make_tessellation(finite_only=False)
        tessellation.build()

        tessellation.add_points(np.array([[0.5, 0.5]]))

        n_native_points = tessellation.points.shape[0]
        n_ghost_points = self._expected_number_of_ghost_points(tessellation)

        self.assertEqual(
            tessellation._boo[:n_native_points],
            [False] * n_native_points,
        )
        self.assertEqual(
            tessellation._boo[n_native_points:],
            [True] * n_ghost_points,
        )

    def test_raise_if_invalid_region_index_rejects_non_integer(self):
        tessellation = self._make_tessellation(finite_only=False)
        tessellation.build()

        with self.assertRaises(TypeError):
            tessellation.raise_if_invalid_region_index(1.2)

    def test_raise_if_invalid_region_index_rejects_out_of_range_index(self):
        tessellation = self._make_tessellation(finite_only=False)
        tessellation.build()

        with self.assertRaises(ValueError):
            tessellation.raise_if_invalid_region_index(len(tessellation.vor.regions))

class AdaptiveSurrogateStudyBaseBehaviorMixin:
    """
    Tests for behavior implemented by AdaptiveSurrogateStudyBase.

    Concrete test classes must provide:

    * ``StudyClass``
    * ``make_study()``
    """

    StudyClass = None

    def make_study(self):
        p1 = Parameter("p1", 0.0, 1.0, 0.5)
        p2 = Parameter("p2", -1.0, 1.0, 0.0)
        return self.StudyClass(p1, p2)

    def test_parameter_bounds(self):
        study = self.make_study()
        bounds = _get_parameter_bounds(study._parameter_collection)
        expected = np.array([[0.0, 1.0], [-1.0, 1.0]])
        np.testing.assert_array_equal(bounds, expected)

    def test_set_independent_variable(self):
        study = self.make_study()
        study.set_independent_variable("time", np.linspace(0, 1, 5))

        self.assertEqual(study._independent_variable, "time")
        np.testing.assert_array_equal(
            study._independent_variable_values,
            np.linspace(0, 1, 5),
        )

        with self.assertRaises(TypeError):
            study.set_independent_variable(1.0, np.linspace(0, 1, 2))

        with self.assertRaises(ValueError):
            study.set_independent_variable("", "")

        with self.assertRaises(TypeError):
            study.set_independent_variable("x", "")

    def test_set_target_field(self):
        study = self.make_study()

        study.set_target_field_name("temperature")
        self.assertEqual(study._target_field_name, "temperature")

        with self.assertRaises(TypeError):
            study.set_target_field_name(1.0)

    def test_set_number_of_test_samples(self):
        study = self.make_study()

        study.set_number_of_test_samples(20)
        self.assertEqual(study._number_of_test_samples, 20)

        with self.assertRaises(TypeError):
            study.set_number_of_test_samples("a")

    def test_set_max_training_samples(self):
        study = self.make_study()

        study.set_max_training_samples(500)
        self.assertEqual(study._max_training_samples, 500)

        with self.assertRaises(TypeError):
            study.set_max_training_samples("a")

    def test_set_test_group_random_seed(self):
        study = self.make_study()

        study.set_seed(10)
        study.set_test_group_random_seed(1234)

        self.assertEqual(study._test_group_random_seed, 1234)
        self.assertEqual(study._seed, 10)

        with self.assertRaises(TypeError):
            study.set_test_group_random_seed("bad")

    def test_default_test_samples_calculation(self):
        study = self.make_study()

        study.set_max_training_samples(200)
        self.assertEqual(study._set_default_number_of_test_samples(), 20)

        study.set_max_training_samples(2000)
        self.assertEqual(study._set_default_number_of_test_samples(), 100)

    def test_make_simulation_results_synchronizer_success(self):
        study = self.make_study()

        study.set_independent_variable("x", [0.0, 1.0])
        study.set_target_field_name("y")

        sync = study._make_simulation_results_synchronizer(None)

        self.assertIsInstance(sync, SimulationResultsSynchronizer)
        self.assertEqual(sync.independent_field, "x")
        self.assertEqual(sync._independent_field_values, [0.0, 1.0])
        self.assertEqual(sync.fields_of_interest, ("y",))

    def test_make_simulation_results_synchronizer_missing_inputs(self):
        study = self.make_study()

        study.set_independent_variable("x", [0, 1])
        with self.assertRaises(RuntimeError) as ctx:
            study._make_simulation_results_synchronizer(None)
        self.assertIn("Target field name", str(ctx.exception))

        study._independent_variable = None
        study._independent_variable_values = None
        study.set_target_field_name("test")

        with self.assertRaises(RuntimeError) as ctx:
            study._make_simulation_results_synchronizer(None)
        self.assertIn("Independent variable name", str(ctx.exception))

    def test_add_evaluation_set_once(self):
        study = self.make_study()

        study.set_independent_variable("t", [0, 1])
        study.set_target_field_name("z")

        self.assertFalse(study._evaluation_set_added)
        study.add_evaluation_set(light_model)
        self.assertTrue(study._evaluation_set_added)

        with self.assertRaises(RuntimeError):
            study.add_evaluation_set(light_model)

    def test_add_evaluation_set_invalid_state(self):
        study = self.make_study()

        study.set_independent_variable("t", [0, 1])
        study.set_target_field_name("z")

        with self.assertRaises(TypeError):
            study.add_evaluation_set(light_model, state="not_a_state")

    def test_results_synchronizer_property_before_and_after_add_evaluation_set(self):
        study = self.make_study()

        self.assertIsNone(study.results_synchronizer)

        study.set_independent_variable("x", [0.0, 1.0])
        study.set_target_field_name("y")
        study.add_evaluation_set(light_model)

        self.assertIsInstance(
            study.results_synchronizer,
            SimulationResultsSynchronizer,
        )

    def test_set_save_filename(self):
        study = self.make_study()

        self.assertIsNone(study.surrogate_save_filename)

        with self.assertRaises(ValueError):
            study.set_surrogate_save_filename("my_surrogate_name")

        with self.assertRaises(TypeError):
            study.set_surrogate_save_filename(0)

        with self.assertRaises(ValueError):
            study.set_surrogate_save_filename("")

        study.set_surrogate_save_filename("my_surrogate_name.joblib")
        self.assertEqual(
            study.surrogate_save_filename,
            "my_surrogate_name.joblib",
        )

    def test_set_surrogate_storage_options(self):
        study = self.make_study()

        self.assertEqual(study._surrogate_storage_best_n_surrogates, 1)
        self.assertIsNone(study._surrogate_storage_every_n_batches)
        self.assertEqual(study._surrogate_storage_score_metric, "max_error")

        study.set_surrogate_storage_options(
            best_n_surrogates=3,
            save_every_n_batches=5,
            score_metric="max_error",
        )

        self.assertEqual(study._surrogate_storage_best_n_surrogates, 3)
        self.assertEqual(study._surrogate_storage_every_n_batches, 5)
        self.assertEqual(study._surrogate_storage_score_metric, "max_error")

        study.set_surrogate_storage_options(
            best_n_surrogates=None,
            save_every_n_batches=2,
            score_metric="score",
        )

        self.assertIsNone(study._surrogate_storage_best_n_surrogates)
        self.assertEqual(study._surrogate_storage_every_n_batches, 2)
        self.assertEqual(study._surrogate_storage_score_metric, "r2")

    def test_set_surrogate_storage_options_invalid_inputs(self):
        study = self.make_study()

        with self.assertRaises(ValueError):
            study.set_surrogate_storage_options(
                best_n_surrogates=None,
                save_every_n_batches=None,
            )

        with self.assertRaises(ValueError):
            study.set_surrogate_storage_options(score_metric="not_a_metric")

        with self.assertRaises(ValueError):
            study.set_surrogate_storage_options(best_n_surrogates=0)

        with self.assertRaises(TypeError):
            study.set_surrogate_storage_options(save_every_n_batches="bad")



class TestVoronoiAdaptiveSurrogateStudyBaseBehavior(
    AdaptiveSurrogateStudyBaseBehaviorMixin,
    MatcalUnitTest,
):
    StudyClass = VoronoiAdaptiveSurrogateStudy

    def setUp(self):
        super().setUp(__file__)
        self.study = self.make_study()

    def _make_two_parameter_study(self):
        return VoronoiAdaptiveSurrogateStudy(
            Parameter("x", 0.0, 1.0),
            Parameter("y", -1.0, 1.0),
        )

    def test_reduce_voronoi_candidates_returns_all_candidates_in_one_at_a_time_mode(self):
        study = self._make_two_parameter_study()
        study._batch_size = 1
        study._thin = 2
        study._random_selection = None

        candidates = np.array([
            [0.0, -1.0],
            [0.25, -0.5],
            [0.5, 0.0],
            [0.75, 0.5],
        ])

        reduced = _reduce_voronoi_candidates(
            candidates,
            study._number_parameters,
            study._batch_size,
            study._thin,
            study._random_selection,
            study._random_generator,
        )

        self.assert_close_arrays(reduced, candidates)

    def test_get_deterministic_in_cell_grid_points_uses_nearest_seed_assignment(self):
        study = self._make_two_parameter_study()
        study.set_selected_cell_search_grid_points(3)

        class FakeVoronoiTessellation:
            points = np.array([
                [0.0, -1.0],
                [1.0, 1.0],
            ])

        study._voronoi_tessellation = FakeVoronoiTessellation()

        seed_location = np.array([0.0, -1.0])
        in_cell_points = study._get_deterministic_in_cell_grid_points(seed_location)

        seed_points = study._voronoi_tessellation.points
        owner_indices = _assign_points_to_nearest_seed(in_cell_points, seed_points)

        self.assertTrue(in_cell_points.shape[0] > 0)
        self.assertTrue(np.all(owner_indices == 0))

    def test_find_first_valid_ranked_candidate_point_returns_first_valid_generated_point(self):
        study = self._make_two_parameter_study()

        candidate_locations = np.array([
            [0.0, -1.0],
            [0.5, 0.0],
            [1.0, 1.0],
        ])

        def fake_find_new_sample_location(location):
            if np.allclose(location, [0.0, -1.0]):
                return None
            if np.allclose(location, [0.5, 0.0]):
                return np.array([0.25, -0.5])
            return np.array([0.75, 0.5])

        study._find_new_sample_location_for_candidate = fake_find_new_sample_location

        point = study._find_first_valid_ranked_candidate_point(candidate_locations)

        self.assert_close_arrays(point, np.array([0.25, -0.5]))

    def test_find_first_valid_ranked_candidate_point_skips_out_of_bounds_points(self):
        study = self._make_two_parameter_study()

        candidate_locations = np.array([
            [0.0, -1.0],
            [0.5, 0.0],
        ])

        proposed_points = [
            np.array([2.0, 0.0]),
            np.array([0.25, -0.5]),
        ]

        def fake_find_new_sample_location(_):
            return proposed_points.pop(0)

        study._find_new_sample_location_for_candidate = fake_find_new_sample_location

        point = study._find_first_valid_ranked_candidate_point(candidate_locations)

        self.assert_close_arrays(point, np.array([0.25, -0.5]))


class TestSparseGridAdaptiveSurrogateStudy(
    AdaptiveSurrogateStudyBaseBehaviorMixin,
    MatcalUnitTest,
):
    StudyClass = SparseGridAdaptiveSurrogateStudy

    def setUp(self):
        super().setUp(__file__)
        self.study = self.make_study()
    
    def test_error_with_add_parameter_set(self):
        with self.assertRaises(RuntimeError):
            self.study.add_parameter_evaluation(p1=1, p2=2)

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
        * treats the sample matrix as native parameter values,
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


class AdaptiveSurrogateActualFitMixin:
    """
    Smoke test that launches an adaptive surrogate study and fits at least one
    surrogate. This intentionally does not assert convergence.
    """

    StudyClass = None

    def make_study(self):
        raise NotImplementedError

    def configure_method_specific_options(self, study):
        pass

    def test_launch_fits_at_least_one_surrogate_without_checking_convergence(self):
        study = self.make_study()

        study.set_independent_variable("x", np.linspace(0.0, 1.0, 8))
        study.set_target_field_name("f")
        study.set_number_of_test_samples(2)
        study.set_max_training_samples(5)
        study.set_error_stopping_criteria(
            rmse_goal=1.0e-30,
            max_abs_error_goal=1.0e-30,
        )
        study.add_evaluation_set(PythonModel(linear_model_2d))

        self.configure_method_specific_options(study)

        results = study.launch()

        self.assertIsNotNone(results)
        self.assertIsNotNone(study.surrogate)
        self.assertGreaterEqual(len(study.surrogate.rmse_history), 1)
        self.assertGreaterEqual(len(study.surrogate.max_error_history), 1)
        self.assertGreaterEqual(len(study.surrogate.sample_count_history), 1)
        self.assertGreaterEqual(len(study.surrogate.surrogate_records), 1)


@unittest.skipIf(
    not HAS_PYAPPROX,
    "pyapprox not installed – skipping sparse-grid fit smoke test",
)
class TestSparseGridAdaptiveSurrogateActualFit(
    AdaptiveSurrogateActualFitMixin,
    MatcalUnitTest,
):
    def setUp(self):
        super().setUp(__file__)

    def make_study(self):
        return SparseGridAdaptiveSurrogateStudy(a, b)

    def configure_method_specific_options(self, study):
        study.set_sparse_grid_adaptivity_limits(max_level=1, pnorm=1.0)
        study.set_surrogate_storage_options(
            best_n_surrogates=1,
            save_every_n_batches=None,
            score_metric="max_error",
        )


class TestVoronoiAdaptiveSurrogateActualFit(
    AdaptiveSurrogateActualFitMixin,
    MatcalUnitTest,
):
    def setUp(self):
        super().setUp(__file__)

    def make_study(self):
        return VoronoiAdaptiveSurrogateStudy(a, b)

    def configure_method_specific_options(self, study):
        study.set_number_of_initial_samples(4)
        study.set_cross_validation_options(
            nsplits=0,
            nmax_folds=1,
            nmax_loo="all",
            cv_metric="sum_abs",
            batch_size=1,
        )
        study.set_voronoi_sampling_options(
            voronoi_type="full",
            finite_only=False,
            iterative_updates=True,
        )
        study.set_surrogate_storage_options(
            best_n_surrogates=1,
            save_every_n_batches=None,
            score_metric="max_error",
        )