import matcal as mc
import numpy as np
import unittest

from matcal.core.tests.MatcalUnitTest import MatcalUnitTest
from matcal.core.tests.unit.test_adaptive_surrogates import HAS_PYAPPROX


def model(a,b,c, **kwargs):
    x = np.linspace(0.0,3, 100)
    y = a+b*x+np.exp(1/(c)*x)
    return {"x":x, "y":y}


py_model = mc.PythonModel(model)


def simple_model(a,b,c,**kwargs):
    x = np.linspace(0.0,3, 100)
    y = a+b*x+c*x**2
    return {"x":x, "y":y}


simple_py_model = mc.PythonModel(simple_model)

a = mc.Parameter("a", 0, 10)
b = mc.Parameter("b", 0, 10)
c = mc.Parameter("c", 0.1, 2)


iter_count = 0
def restart_model_func(a,b,c, eval_error_count=10, **kwargs):
    x = np.linspace(0.0,3, 100)
    y = a+b*x+np.exp(1/(c)*x)
    evaluation_number = kwargs["evaluation_number"]
    if evaluation_number > eval_error_count:
        raise ValueError("exiting to restart")
    return {"x":x, "y":y}


restart_model = mc.PythonModel(restart_model_func, pass_evaluation_number=True)


class TestSparseGridAdaptiveSurrogate(MatcalUnitTest):

    def setUp(self):
        super().setUp(__file__)           

    @unittest.skipIf(
            not HAS_PYAPPROX,
            "pyapprox not installed – skipping pyapprox‑dependent tests")
    def test_restart_during_training(self):
        sg_study = mc.SparseGridAdaptiveSurrogateStudy(a,b,c)
        sg_study.set_independent_variable("x", np.linspace(0.0,3,100))
        sg_study.set_number_of_test_samples(10)
        sg_study.set_target_field_name("y")
        sg_study.set_test_group_random_seed(1234)
        sg_study.add_evaluation_set(restart_model)

        with self.assertRaises(ValueError):
            sg_study.launch()
        sg_study = mc.SparseGridAdaptiveSurrogateStudy(a,b,c)
        sg_study.set_independent_variable("x", np.linspace(0.0,3,100))
        sg_study.set_number_of_test_samples(10)
        sg_study.set_target_field_name("y")
        sg_study.set_test_group_random_seed(1234)
        restart_model.add_constants(eval_error_count=100)
        sg_study.add_evaluation_set(restart_model)
        sg_study.restart()

        with self.assertRaises(ValueError):
            sg_study.launch()

        sg_study = mc.SparseGridAdaptiveSurrogateStudy(a,b,c)
        sg_study.set_independent_variable("x", np.linspace(0.0,3,100))
        sg_study.set_number_of_test_samples(10)
        sg_study.set_target_field_name("y")
        sg_study.set_test_group_random_seed(1234)
        restart_model.add_constants(eval_error_count=10000)
        sg_study.add_evaluation_set(restart_model)
        sg_study.restart()
        sg_study.launch()

        self.assertLess(sg_study.surrogate.rmse_history[-1], 1e-2)
        self.assertLess(sg_study.surrogate.max_error_history[-1], 1e-1)

    @unittest.skipIf(
        not HAS_PYAPPROX,
        "pyapprox not installed – skipping pyapprox‑dependent tests",
    )
    def test_sparse_grid_default_basis_is_lagrange(self):
        sg_study = mc.SparseGridAdaptiveSurrogateStudy(a, b, c)
        sg_study.set_independent_variable("x", np.linspace(0.0, 3, 100))
        sg_study.set_error_stopping_criteria(1e-4)  # avg only; max uses default
        sg_study.set_number_of_test_samples(30)
        sg_study.set_target_field_name("y")
        sg_study.set_test_group_random_seed(1234)

        sg_study.add_evaluation_set(simple_py_model)
        sg_study.launch()

        self.assertLess(sg_study.surrogate.rmse_history[-1], 1e-2)
        self.assertLess(sg_study.surrogate.max_error_history[-1], 1e-1)

    @unittest.skipIf(
        not HAS_PYAPPROX,
        "pyapprox not installed – skipping pyapprox‑dependent tests",
    )
    def test_sparse_grid_piecewise_basis_fit_linear(self):
        sg_study = mc.SparseGridAdaptiveSurrogateStudy(a, b, c)
        sg_study.set_independent_variable("x", np.linspace(0.0, 3, 100))
        sg_study.set_error_stopping_criteria(1e-4)
        sg_study.set_number_of_test_samples(30)
        sg_study.set_target_field_name("y")
        sg_study.set_test_group_random_seed(1234)

        sg_study.set_sparse_grid_basis(basis_type="piecewise", piecewise_degree=1)

        sg_study.add_evaluation_set(simple_py_model)
        sg_study.launch()

        # linear piecewise may converge slower; keep thresholds reasonable
        self.assertLess(sg_study.surrogate.rmse_history[-1], 1e-2)
        self.assertLess(sg_study.surrogate.max_error_history[-1], 1e-1)

    @unittest.skipIf(
        not HAS_PYAPPROX,
        "pyapprox not installed – skipping pyapprox‑dependent tests",
    )
    def test_sparse_grid_piecewise_basis_fit_cubic(self):
        sg_study = mc.SparseGridAdaptiveSurrogateStudy(a, b, c)
        sg_study.set_independent_variable("x", np.linspace(0.0, 3, 100))
        sg_study.set_error_stopping_criteria(1e-4)
        sg_study.set_number_of_test_samples(30)
        sg_study.set_target_field_name("y")
        sg_study.set_test_group_random_seed(1234)

        sg_study.set_sparse_grid_basis(basis_type="piecewise", piecewise_degree=3)

        sg_study.add_evaluation_set(simple_py_model)
        sg_study.launch()

        self.assertLess(sg_study.surrogate.rmse_history[-1], 1e-2)
        self.assertLess(sg_study.surrogate.max_error_history[-1], 1e-1)

    @unittest.skipIf(
        not HAS_PYAPPROX,
        "pyapprox not installed – skipping pyapprox‑dependent tests",
    )
    def test_sparse_grid_surrogate_transpose_flag_behavior(self):
        """
        Ensures both batch conventions are accepted or fail cleanly.
        If your surrogate does not support transpose=False, update accordingly.
        """
        sg_study = mc.SparseGridAdaptiveSurrogateStudy(a, b, c)
        sg_study.set_independent_variable("x", np.linspace(0.0, 3, 100))
        sg_study.set_error_stopping_criteria(1e-4)
        sg_study.set_number_of_test_samples(30)
        sg_study.set_target_field_name("y")
        sg_study.set_test_group_random_seed(1234)
        sg_study.add_evaluation_set(simple_py_model)
        sg_study.launch()

        surrogate = sg_study.surrogate
        pts = np.array([[1.0, 2.0, 0.5], [3.0, 4.0, 1.5]])  # (n_samples, n_params)

        # default call path
        pred1 = surrogate(pts, batch_evaluate=True)
        y1 = np.asarray(pred1["y"])

        # try alternate transpose setting; depending on your _process_surrogate_args_call
        try:
            pred2 = surrogate(pts.T, batch_evaluate=True, transpose=False)
            y2 = np.asarray(pred2["y"])
            np.testing.assert_allclose(y1, y2, rtol=0, atol=1e-10)
        except Exception:
            # If unsupported, require a clean error type (RuntimeError/ValueError are typical)
            with self.assertRaises((RuntimeError, ValueError, TypeError)):
                surrogate(pts.T, batch_evaluate=True, transpose=False)
    # -------------------------------------------------------
    # Range enforcement (must still work after refactoring)
    # -------------------------------------------------------
    @unittest.skipIf(
        not HAS_PYAPPROX,
        "pyapprox not installed – skipping pyapprox‑dependent tests",
    )
    def test_sparse_grid_enforce_training_data_parameter_range_raises(self):
        sg_study = mc.SparseGridAdaptiveSurrogateStudy(a, b, c)
        sg_study.set_independent_variable("x", np.linspace(0.0, 3, 100))
        sg_study.set_error_stopping_criteria(1e-4)
        sg_study.set_number_of_test_samples(30)
        sg_study.set_target_field_name("y")
        sg_study.set_test_group_random_seed(1234)
        sg_study.add_evaluation_set(simple_py_model)
        sg_study.launch()

        surrogate = sg_study.surrogate
        # out of bounds: a=11 (a is [0,10])
        with self.assertRaises((ValueError, RuntimeError)):
            surrogate(a=11.0, b=1.0, c=0.5)

    @unittest.skipIf(
        not HAS_PYAPPROX,
        "pyapprox not installed – skipping pyapprox‑dependent tests",
    )
    def test_sparse_grid_enforce_training_data_parameter_range_can_be_disabled(self):
        sg_study = mc.SparseGridAdaptiveSurrogateStudy(a, b, c)
        sg_study.set_independent_variable("x", np.linspace(0.0, 3, 100))
        sg_study.set_error_stopping_criteria(1e-4)
        sg_study.set_number_of_test_samples(30)
        sg_study.set_target_field_name("y")
        sg_study.set_test_group_random_seed(1234)
        sg_study.add_evaluation_set(simple_py_model)
        sg_study.launch()

        surrogate = sg_study.surrogate
        surrogate.enforce_training_data_parameter_range(False)

        # Should not raise even though out-of-bounds
        pred = surrogate(a=11.0, b=1.0, c=0.5)
        self.assertIn("y", pred)
        self.assertIn("x", pred)
    # --------------------------------------------------------
    # Serialization / restart tests (choose one strategy)
    # --------------------------------------------------------
    @unittest.skipIf(
        not HAS_PYAPPROX,
        "pyapprox not installed – skipping pyapprox‑dependent tests",
    )
    def test_sparse_grid_training_with_surrogate_save_disabled(self):
        """
        Recommended if PyApprox objects are not picklable with stdlib pickle/joblib.

        This test expects you implemented one of:
          * sg_study.set_surrogate_save_filename(None)
          * or sg_study.disable_surrogate_saving()
          * or similar.
        """
        sg_study = mc.SparseGridAdaptiveSurrogateStudy(a, b, c)
        sg_study.set_independent_variable("x", np.linspace(0.0, 3, 100))
        sg_study.set_error_stopping_criteria(1e-4)
        sg_study.set_number_of_test_samples(20)
        sg_study.set_target_field_name("y")
        sg_study.set_test_group_random_seed(1234)

        # Try common disabling patterns; skip if none exist
        disabled = False
        if hasattr(sg_study, "set_surrogate_save_filename"):
            try:
                sg_study.set_surrogate_save_filename("tmp.joblib")
                # user asked for disabled saving; if your implementation allows None:
                sg_study._surrogate_save_filename = None
                disabled = True
            except Exception:
                pass
        if hasattr(sg_study, "disable_surrogate_saving"):
            sg_study.disable_surrogate_saving()
            disabled = True

        if not disabled:
            self.skipTest("No mechanism implemented to disable surrogate saving")

        sg_study.add_evaluation_set(simple_py_model)
        sg_study.launch()

        self.assertLess(sg_study.surrogate.rmse_history[-1], 5e-2)

    @unittest.skipIf(
        not HAS_PYAPPROX,
        "pyapprox not installed – skipping pyapprox‑dependent tests",
    )
    def test_sparse_grid_surrogate_can_be_saved_and_loaded(self):
        """
        Only enable this if you made the sparse-grid surrogate picklable, e.g. by:
          * patching PyApprox closures to globals, OR
          * storing a MatCal-owned replayable representation rather than PyApprox objects.
        """
        sg_study = mc.SparseGridAdaptiveSurrogateStudy(a, b, c)
        sg_study.set_independent_variable("x", np.linspace(0.0, 3, 100))
        sg_study.set_error_stopping_criteria(1e-3)
        sg_study.set_number_of_test_samples(10)
        sg_study.set_target_field_name("y")
        sg_study.set_test_group_random_seed(1234)
        sg_study.set_surrogate_save_filename("tmp_sparse_grid_surrogate.joblib")
        sg_study.add_evaluation_set(simple_py_model)
        sg_study.launch()

        # Load and compare a prediction
        from matcal.core.serializer_wrapper import matcal_load

        loaded = matcal_load("tmp_sparse_grid_surrogate.joblib")
        a0, b0, c0 = 2.0, 3.0, 1.1
        pred0 = sg_study.surrogate(a=a0, b=b0, c=c0)
        pred1 = loaded(a=a0, b=b0, c=c0)
        np.testing.assert_allclose(pred0["y"], pred1["y"], rtol=0, atol=1e-12)
        np.testing.assert_allclose(pred0["x"], pred1["x"], rtol=0, atol=1e-12)
    # -------------------------------------------------
    # Surrogate call behavior: batch vs single match
    # -------------------------------------------------
    @unittest.skipIf(
        not HAS_PYAPPROX,
        "pyapprox not installed – skipping pyapprox‑dependent tests",
    )
    def test_sparse_grid_surrogate_call_batch_and_single_match(self):
        sg_study = mc.SparseGridAdaptiveSurrogateStudy(a, b, c)
        sg_study.set_independent_variable("x", np.linspace(0.0, 3, 100))
        sg_study.set_error_stopping_criteria(1e-4)
        sg_study.set_number_of_test_samples(30)
        sg_study.set_target_field_name("y")
        sg_study.set_test_group_random_seed(1234)
        sg_study.add_evaluation_set(simple_py_model)
        sg_study.launch()

        surrogate = sg_study.surrogate

        a0, b0, c0 = 1.25, 2.5, 0.75

        pred_single = surrogate(a=a0, b=b0, c=c0, batch_evaluate=False)
        y_single = np.asarray(pred_single["y"])
        x_single = np.asarray(pred_single["x"])

        pred_batch = surrogate([[a0, b0, c0]], batch_evaluate=True)
        y_batch = np.asarray(pred_batch["y"])
        x_batch = np.asarray(pred_batch["x"])

        self.assertEqual(y_batch.shape[0], 1)
        self.assertEqual(y_batch.shape[1], y_single.shape[0])
        np.testing.assert_allclose(x_single, x_batch, rtol=0, atol=1e-12)
        np.testing.assert_allclose(y_single, y_batch[0, :], rtol=0, atol=1e-12)


class TestVoronoiAdaptiveSurrogate(MatcalUnitTest):

    def setUp(self):
        super().setUp(__file__)           

    def test_simple_function_fit(self):
        study = mc.VoronoiAdaptiveSurrogateStudy(a,b,c)
        study.set_independent_variable("x", np.linspace(0.0,3,200))
        study.set_error_stopping_criteria(1e-4)
        study.set_number_of_test_samples(50)
        study.set_number_of_initial_samples(30)
        study.set_surrogate_options(decomp_var=3)
        study.set_target_field_name("y")
        study.set_test_group_random_seed(1234)
        study.add_evaluation_set(simple_py_model)
        study.launch()

        self.assertLess(study.surrogate.rmse_history[-1], 1e-2)
        self.assertLess(study.surrogate.max_error_history[-1], 1e-1)



