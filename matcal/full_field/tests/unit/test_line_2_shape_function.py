from matcal.full_field.shapefunctions import OneDim2NodeLinearShapeFunction
from matcal.core.tests.MatcalUnitTest import MatcalUnitTest
import numpy as np


class TestLine2ShapeFunction(MatcalUnitTest):

    def setUp(self):
        super().setUp(__file__)
        self.sf = OneDim2NodeLinearShapeFunction()

    def test_zero_value_return_one_half(self):
        x = np.zeros(1)
        value = self.sf.values(x)
        self.assertTrue(np.allclose(value, np.ones(2) * .5))

    def test_three_zeros_return_three_one_halfs(self):
        x = np.zeros(3)
        values = self.sf.values(x)
        self.assertTrue(np.allclose(values, np.ones([2, 3]) / 2))

    def test_range_points(self):
        x = np.linspace(-1, 1, 20)
        v1 = -.5 * (x - 1)
        v2 = .5 * (x + 1)
        self.assertTrue(np.allclose(self.sf.values(x), np.array([v1, v2])))

    def test_zero_get_grad_of_one_half(self):
        x = np.zeros(1)
        grad = self.sf.gradients(x)
        goal_grad = np.array([[-.5], [.5]])
        self.assertTrue(np.allclose(grad, goal_grad))

    def test_range_grad_return_one_half(self):
        x = np.linspace(-1, 1, 20)
        g1 = np.ones(20) * -.5
        g2 = np.ones(20) * .5
        goal_grad = np.array([g1,g2])
        self.assertTrue(np.allclose(self.sf.gradients(x), goal_grad))


    def test_get_number_of_shape_functions(self):
        self.assertEqual(self.sf.number_of_functions, 2)