import os
from contextlib import contextmanager

import matcal as mc
from matcal.core.tests.MatcalUnitTest import MatcalUnitTest
import numpy as np


_RESTART_POISON_ENV = "MATCAL_TEST_PARAMETER_STUDY_RESTART_POISON_MODEL"


def quad(a, b):
    n_pts = 30
    x = np.linspace(0, 1, n_pts)
    y = b * x + a * np.power(x, 2)
    import time
    time.sleep(0.1)
    return x, y


def model_wrapper(**params):
    if os.environ.get(_RESTART_POISON_ENV) == "1":
        x = np.linspace(0, 1, 30)
        y = np.full_like(x, 1.0e100)
        return {"x": x, "y": y}

    a = params["a"]
    b = params["b"]
    out = quad(a, b)
    return {"x": out[0], "y": out[1]}


@contextmanager
def poison_model_if_rerun():
    old_value = os.environ.get(_RESTART_POISON_ENV)
    os.environ[_RESTART_POISON_ENV] = "1"
    try:
        yield
    finally:
        if old_value is None:
            os.environ.pop(_RESTART_POISON_ENV, None)
        else:
            os.environ[_RESTART_POISON_ENV] = old_value


class TestParameterStudy(MatcalUnitTest):

    def setUp(self):
        super().setUp(__file__)

    def _assert_restart_results_match(self, results, restart_results, model):
        self.assertTrue(results.success, results.exit_message)
        self.assertTrue(restart_results.success, restart_results.exit_message)

        self.assertEqual(
            results.number_of_evaluations,
            restart_results.number_of_evaluations,
        )

        self.assertEqual(
            results.evaluation_ids,
            restart_results.evaluation_ids,
        )

        np.testing.assert_allclose(
            results.total_objective_history,
            restart_results.total_objective_history,
        )

        self.assertEqual(
            list(results.parameter_history.keys()),
            list(restart_results.parameter_history.keys()),
        )

        for param_name in results.parameter_history:
            np.testing.assert_allclose(
                results.parameter_history[param_name],
                restart_results.parameter_history[param_name],
            )

        base_hist = results.simulation_history[model.name]["matcal_default_state"]
        restart_hist = restart_results.simulation_history[model.name]["matcal_default_state"]

        self.assertEqual(len(base_hist), len(restart_hist))

        for base, restart in zip(base_hist, restart_hist):
            self.assert_close_dicts_or_data(base, restart)

    def test_restart_parameter_study_gets_same_answers_serial(self):
        a = mc.Parameter("a", 0, 1)
        b = mc.Parameter("b", 0, 1)
        pc = mc.ParameterCollection("ab", a, b)

        model = mc.PythonModel(model_wrapper)

        x_ref = np.linspace(0, 1, 10)
        obj = mc.SimulationResultsSynchronizer("x", x_ref, "y")

        n_evals = 20
        rng = np.random.default_rng(12345)
        eval_dict = {
            "a": rng.uniform(0, 1, n_evals),
            "b": rng.uniform(0, 1, n_evals),
        }

        study = mc.ParameterStudy(pc)
        study.run_in_serial()
        study.add_evaluation_set(model, obj)

        for i_eval in range(n_evals):
            a_val = eval_dict["a"][i_eval]
            b_val = eval_dict["b"][i_eval]
            study.add_parameter_evaluation(a=a_val, b=b_val)

        results = study.launch()

        study2 = mc.ParameterStudy(pc)
        study2.run_in_serial()
        study2.add_evaluation_set(model, obj)
        study2.restart()

        for i_eval in range(n_evals):
            a_val = eval_dict["a"][i_eval]
            b_val = eval_dict["b"][i_eval]
            study2.add_parameter_evaluation(a=a_val, b=b_val)

        with poison_model_if_rerun():
            restart_results = study2.launch()

        self._assert_restart_results_match(results, restart_results, model)

    def test_restart_parameter_study_gets_same_answers_distributed(self):
        a = mc.Parameter("a", 0, 1)
        b = mc.Parameter("b", 0, 1)
        pc = mc.ParameterCollection("ab", a, b)

        model = mc.PythonModel(model_wrapper)

        x_ref = np.linspace(0, 1, 10)
        obj = mc.SimulationResultsSynchronizer("x", x_ref, "y")

        n_evals = 50
        rng = np.random.default_rng(12345)
        eval_dict = {
            "a": rng.uniform(0, 1, n_evals),
            "b": rng.uniform(0, 1, n_evals),
        }

        study = mc.ParameterStudy(pc)
        study.add_evaluation_set(model, obj)

        for i_eval in range(n_evals):
            a_val = eval_dict["a"][i_eval]
            b_val = eval_dict["b"][i_eval]
            study.add_parameter_evaluation(a=a_val, b=b_val)

        results = study.launch()

        study2 = mc.ParameterStudy(pc)
        study2.add_evaluation_set(model, obj)
        study2.restart()

        for i_eval in range(n_evals):
            a_val = eval_dict["a"][i_eval]
            b_val = eval_dict["b"][i_eval]
            study2.add_parameter_evaluation(a=a_val, b=b_val)

        with poison_model_if_rerun():
            restart_results = study2.launch()

        self._assert_restart_results_match(results, restart_results, model)

    def test_restart_parameter_study_gets_same_answers_distributed_parallel(self):
        a = mc.Parameter("a", 0, 1)
        b = mc.Parameter("b", 0, 1)
        pc = mc.ParameterCollection("ab", a, b)

        model = mc.PythonModel(model_wrapper)

        x_ref = np.linspace(0, 1, 10)
        obj = mc.SimulationResultsSynchronizer("x", x_ref, "y")

        n_evals = 50
        rng = np.random.default_rng(12345)
        eval_dict = {
            "a": rng.uniform(0, 1, n_evals),
            "b": rng.uniform(0, 1, n_evals),
        }

        study = mc.ParameterStudy(pc)
        study.add_evaluation_set(model, obj)
        n_cores = 4
        study.set_core_limit(n_cores)

        for i_eval in range(n_evals):
            a_val = eval_dict["a"][i_eval]
            b_val = eval_dict["b"][i_eval]
            study.add_parameter_evaluation(a=a_val, b=b_val)

        results = study.launch()

        study2 = mc.ParameterStudy(pc)
        study2.add_evaluation_set(model, obj)
        study2.set_core_limit(n_cores)
        study2.restart()

        for i_eval in range(n_evals):
            a_val = eval_dict["a"][i_eval]
            b_val = eval_dict["b"][i_eval]
            study2.add_parameter_evaluation(a=a_val, b=b_val)

        with poison_model_if_rerun():
            restart_results = study2.launch()

        self._assert_restart_results_match(results, restart_results, model)

    def test_restart_parameter_study_raise_error_for_parallel_threads(self):
        a = mc.Parameter("a", 0, 1)
        b = mc.Parameter("b", 0, 1)
        pc = mc.ParameterCollection("ab", a, b)

        model = mc.PythonModel(model_wrapper)

        x_ref = np.linspace(0, 1, 10)
        obj = mc.SimulationResultsSynchronizer("x", x_ref, "y")

        n_cores = 2
        n_evals = 50
        rng = np.random.default_rng(12345)
        eval_dict = {
            "a": rng.uniform(0, 1, n_evals),
            "b": rng.uniform(0, 1, n_evals),
        }

        study = mc.ParameterStudy(pc)
        study.add_evaluation_set(model, obj)
        study.set_core_limit(n_cores)
        study.set_use_threads()

        for i_eval in range(n_evals):
            a_val = eval_dict["a"][i_eval]
            b_val = eval_dict["b"][i_eval]
            study.add_parameter_evaluation(a=a_val, b=b_val)

        study.launch()

        study2 = mc.ParameterStudy(pc)
        study2.add_evaluation_set(model, obj)
        study2.set_core_limit(n_cores)
        study2.set_use_threads()
        study2.restart()

        with self.assertRaises(RuntimeError):
            study2.launch()