from matcal.full_field.shapefunctions import TwoDim4NodeBilinearShapeFunction
from matcal.core.tests.MatcalUnitTest import MatcalUnitTest
import numpy as np


class TestQuad4ShapeFunction(MatcalUnitTest):

    def setUp(self):
        super().setUp(__file__)
        self.sf = TwoDim4NodeBilinearShapeFunction()

    def test_zero_return_one_quarter(self):
        x = np.zeros([1, 2])
        goal_values = np.ones([1, 4]) * .25
        values = self.sf.values(x)
        self.assertTrue(np.allclose(values, goal_values))

    def test_lower_left_corner_1_0_0_0(self):
        x = np.array([[-1, -1]])
        goal_values = np.array([1, 0, 0, 0])
        values = self.sf.values(x)
        self.assertTrue(np.allclose(values, goal_values))

    def test_lower_right_corner_0_1_0_0(self):
        x = np.array([[1, -1]])
        goal_values = np.array([[0, 1, 0, 0]])
        values = self.sf.values(x)
        self.assertTrue(np.allclose(values, goal_values))

    def test_upper_right_corner_0_0_1_0(self):
        x = np.array([[1, 1]])
        goal_values = np.array([[0, 0, 1, 0]])
        values = self.sf.values(x)
        self.assertTrue(np.allclose(values, goal_values))

    def test_upper_left_corner_0_0_0_1(self):
        x = np.array([[-1, 1]])
        goal_values = np.array([[0, 0, 0, 1]])
        values = self.sf.values(x)
        self.assertTrue(np.allclose(values, goal_values))

    def test_get_corners_and_center(self):
        x = np.array([[0, 0], [-1, -1], [1, -1], [1, 1], [-1, 1]])
        goal_values = np.array([[.25, .25, .25, .25],
                                [1, 0, 0, 0],
                                [0, 1, 0, 0],
                                [0, 0, 1, 0],
                                [0, 0, 0, 1]])
        values = self.sf.values(x)
        self.assertTrue(np.allclose(values, goal_values))

    def test_confirm_gradient_at_center(self):
        x = np.array([[0, 0]])
        goal_values = np.zeros([1, 2, 4])
        goal_values[0, 0, :] = [-.25, .25, .25, -.25]
        goal_values[0, 1, :] = [-.25, -.25, .25, .25]
        values = self.sf.gradients(x)
        self.assertTrue(np.allclose(values, goal_values))

    def test_confirm_gradient_lower_left_corner(self):
        x = np.array([[-1, -1]])
        goal_values = np.zeros([1, 2, 4])
        goal_values[0, 0, :] = [-.5, .5, .0, .0]
        goal_values[0, 1, :] = [-.5, .0, .0, .5]
        values = self.sf.gradients(x)
        self.assertTrue(np.allclose(values, goal_values))

    def test_confirm_gradient_lower_right_corner(self):
        x = np.array([[1, -1]])
        goal_values = np.zeros([1, 2, 4])
        goal_values[0, 0, :] = [-.5, .5, .0, .0]
        goal_values[0, 1, :] = [.0, -.5, .5, .0]
        values = self.sf.gradients(x)
        self.assertTrue(np.allclose(values, goal_values))

    def test_confirm_gradient_upper_right_corner(self):
        x = np.array([[1, 1]])
        goal_values = np.zeros([1, 2, 4])
        goal_values[0, 0, :] = [.0, .0, .5, -.5]
        goal_values[0, 1, :] = [.0, -.5, .5, .0]
        values = self.sf.gradients(x)
        self.assertTrue(np.allclose(values, goal_values))

    def test_confirm_gradient_upper_left_corner(self):
        x = np.array([[-1, 1]])
        goal_values = np.zeros([1, 2, 4])
        goal_values[0, 0, :] = [.0, .0, .5, -.5]
        goal_values[0, 1, :] = [-.5, .0, .0, .5]
        values = self.sf.gradients(x)
        self.assertTrue(np.allclose(values, goal_values))

    def test_confirm_gradient_multiple_points(self):
        x = np.array([[0, 0], [-1, -1], [1, -1], [1, 1], [-1, 1]])
        goal_values = np.zeros([5, 2, 4])
        goal_values[0, 0, :] = [-.25, .25, .25, -.25]
        goal_values[0, 1, :] = [-.25, -.25, .25, .25]
        goal_values[1, 0, :] = [-.5, .5, .0, .0]
        goal_values[1, 1, :] = [-.5, .0, .0, .5]
        goal_values[2, 0, :] = [-.5, .5, .0, .0]
        goal_values[2, 1, :] = [.0, -.5, .5, .0]
        goal_values[3, 0, :] = [.0, .0, .5, -.5]
        goal_values[3, 1, :] = [.0, -.5, .5, .0]
        goal_values[4, 0, :] = [.0, .0, .5, -.5]
        goal_values[4, 1, :] = [-.5, .0, .0, .5]
        values = self.sf.gradients(x)
        self.assertTrue(np.allclose(values, goal_values))

    def test_get_number_of_shape_functions(self):
        self.assertEqual(self.sf.number_of_functions, 4)