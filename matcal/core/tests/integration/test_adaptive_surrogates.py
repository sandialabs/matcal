import matcal as mc
import numpy as np
import unittest

from matcal.core.tests.MatcalUnitTest import MatcalUnitTest
from matcal.core.tests.unit.test_adaptive_surrogates import HAS_PYAPPROX


def model(a,b,c, **kwargs):
    x = np.linspace(0.1,3, 100)
    y = a+b*x+np.exp(1/(c)*x)
    return {"x":x, "y":y}


py_model = mc.PythonModel(model)


a = mc.Parameter("a", 0, 10)
b = mc.Parameter("b", 0, 10)
c = mc.Parameter("c", 0.1, 2)


iter_count = 0
def restart_model_func(a,b,c, eval_error_count=10, **kwargs):
    x = np.linspace(0.1,3, 100)
    y = a+b*x+np.exp(1/(c)*x)
    evaluation_number = kwargs["evaluation_number"]
    if evaluation_number > eval_error_count:
        raise ValueError("exiting to restart")
    return {"x":x, "y":y}


restart_model = mc.PythonModel(restart_model_func, pass_evaluation_number=True)


class TestSparseGridAdaptiveSurrogate(MatcalUnitTest):

    def setUp(self):
        super().setUp(__file__)           

    @unittest.skipIf(not HAS_PYAPPROX,
                 "pyapprox not installed – skipping pyapprox‑dependent tests")
    def test_simple_function_fit(self):
        sg_study = mc.SparseGridAdaptiveSurrogateStudy(a,b,c)
        sg_study.set_independent_variable("x", np.linspace(0,3,100))
        sg_study.set_number_of_test_samples(50)
        sg_study.set_target_field_name("y")
        sg_study.set_test_group_random_seed(1234)
        sg_study.add_evaluation_set(py_model)
        sg_study.launch()

        self.assertLess(sg_study.surrogate.average_error_history[-1], 1e-2)
        self.assertLess(sg_study.surrogate.max_error_history[-1], 1e-1)

        

    @unittest.skipIf(not HAS_PYAPPROX,
                 "pyapprox not installed – skipping pyapprox‑dependent tests")
    def test_restart_during_training(self):
        sg_study = mc.SparseGridAdaptiveSurrogateStudy(a,b,c)
        sg_study.set_independent_variable("x", np.linspace(0,3,100))
        sg_study.set_number_of_test_samples(10)
        sg_study.set_target_field_name("y")
        sg_study.set_test_group_random_seed(1234)
        sg_study.add_evaluation_set(restart_model)

        with self.assertRaises(ValueError):
            sg_study.launch()
        sg_study = mc.SparseGridAdaptiveSurrogateStudy(a,b,c)
        sg_study.set_independent_variable("x", np.linspace(0,3,100))
        sg_study.set_number_of_test_samples(10)
        sg_study.set_target_field_name("y")
        sg_study.set_test_group_random_seed(1234)
        restart_model.add_constants(eval_error_count=100)
        sg_study.add_evaluation_set(restart_model)
        sg_study.restart()

        with self.assertRaises(ValueError):
            sg_study.launch()

        sg_study = mc.SparseGridAdaptiveSurrogateStudy(a,b,c)
        sg_study.set_independent_variable("x", np.linspace(0,3,100))
        sg_study.set_number_of_test_samples(10)
        sg_study.set_target_field_name("y")
        sg_study.set_test_group_random_seed(1234)
        restart_model.add_constants(eval_error_count=10000)
        sg_study.add_evaluation_set(restart_model)
        sg_study.restart()
        sg_study.launch()

        self.assertLess(sg_study.surrogate.average_error_history[-1], 1e-2)
        self.assertLess(sg_study.surrogate.max_error_history[-1], 1e-1)






