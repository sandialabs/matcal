from abc import ABC, abstractmethod
import numpy as np


class ShapeFunctionBase(ABC):

    @abstractmethod
    def values(self, x_array):
        """"""

    @abstractmethod
    def gradients(self, x_array):
        """"""

    @property
    @abstractmethod
    def number_of_functions(self):
        """"""


class OneDim2NodeLinearShapeFunction(ShapeFunctionBase):
    _number_of_functions = 2

    def values(self, x_array):
        return np.array([-(x_array - 1) / 2, (x_array + 1) / 2])

    def gradients(self, x_array):
        grad = np.ones([self._number_of_functions, len(x_array)])
        grad[0, :] = grad[0, :] / (-2)
        grad[1, :] = grad[1, :] / 2
        return grad

    @property
    def number_of_functions(self):
        return self._number_of_functions


class TwoDim4NodeBilinearShapeFunction(ShapeFunctionBase):
    _line_element = OneDim2NodeLinearShapeFunction()
    _number_of_functions = 4

    def values(self, x_array):
        one_d_values_x, one_d_values_y = self._get_1d_values(x_array)
        phi_1 = one_d_values_x[0, :] * one_d_values_y[0, :]
        phi_2 = one_d_values_x[1, :] * one_d_values_y[0, :]
        phi_3 = one_d_values_x[1, :] * one_d_values_y[1, :]
        phi_4 = one_d_values_x[0, :] * one_d_values_y[1, :]
        return np.array([phi_1, phi_2, phi_3, phi_4]).T

    def gradients(self, x_array):
        one_d_values_x, one_d_values_y = self._get_1d_values(x_array)
        one_d_grad_x, one_d_grad_y = self._get_1d_gradients(x_array)
        grad_x = self._create_x_gradient(one_d_values_y, one_d_grad_x)
        grad_y = self._create_y_gradient(one_d_values_x, one_d_grad_y)
        return np.transpose(np.array([grad_x, grad_y]), axes=(2, 0, 1))

    @property
    def number_of_functions(self):
        return self._number_of_functions

    def _get_1d_values(self, x_array):
        one_d_values_x = self._line_element.values(x_array[:, 0])
        one_d_values_y = self._line_element.values(x_array[:, 1])
        return one_d_values_x, one_d_values_y

    def _get_1d_gradients(self, x_array):
        one_d_grad_x = self._line_element.gradients(x_array[:, 0])
        one_d_grad_y = self._line_element.gradients(x_array[:, 1])
        return one_d_grad_x, one_d_grad_y

    def _create_y_gradient(self, value_x, grad_y):
        dphi_1 = value_x[0, :] * grad_y[0, :]
        dphi_2 = value_x[1, :] * grad_y[0, :]
        dphi_3 = value_x[1, :] * grad_y[1, :]
        dphi_4 = value_x[0, :] * grad_y[1, :]
        return np.array([dphi_1, dphi_2, dphi_3, dphi_4])

    def _create_x_gradient(self, value_y, grad_x):
        dphi_1 = grad_x[0, :] * value_y[0, :]
        dphi_2 = grad_x[1, :] * value_y[0, :]
        dphi_3 = grad_x[1, :] * value_y[1, :]
        dphi_4 = grad_x[0, :] * value_y[1, :]
        return np.array([dphi_1, dphi_2, dphi_3, dphi_4])
