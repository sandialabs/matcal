from abc import ABC, abstractmethod
import numpy as np
import scipy.special as special


class NegativePolynomialOrderError(RuntimeError):

    def __init__(self, order):
        message = f"\n  Polynomial orders must be => 0.\n  Passed order: {order}"
        super().__init__(message)


def _check_nonnegative_polynomial_power(order):
    if order < 0:
        raise NegativePolynomialOrderError(order)

def generate_full_one_dim_polynomial_powers(poly_order):
    _check_nonnegative_polynomial_power(poly_order)
    return np.array(list(range(poly_order + 1))).reshape([1, -1])

def generate_mid_and_edge_two_dim_polynimial_powers(poly_order):
    _check_nonnegative_polynomial_power(poly_order)
    n_entries = entries_count_for_mid_and_edge_polynomial(poly_order)
    offset = 1 
    n_dim =2
    poly = np.zeros([2, n_entries], dtype=int)
    for current_order in range(1, poly_order+1):
        for dim_i in range(n_dim):
            poly[dim_i, offset] = current_order
            offset += 1
        if current_order%2 == 0:
            poly[:, offset] = np.ones(n_dim) * (current_order//2)
            offset += 1
    return poly

def entries_count_for_mid_and_edge_polynomial(poly_order):
    n_entries = (poly_order//2) * 5 + (poly_order%2) * 2 +1
    return n_entries


def generate_full_two_dim_polynomial_powers(poly_order):
    _check_nonnegative_polynomial_power(poly_order)
    n_entries = entries_count_for_full_two_dim_polynomial(poly_order)
    poly = np.zeros([2, n_entries], dtype=int)
    offset = 1
    for current_order in range(1, poly_order + 1):
        for y_power in range(current_order + 1):
            poly[0, offset] = current_order - y_power
            poly[1, offset] = y_power
            offset += 1
    return poly

def generate_nocross_two_dim_polynomial_powers(poly_order):
    _check_nonnegative_polynomial_power(poly_order)
    n_dim = 2
    poly = entries_count_for_no_cross_ndim_polynomial(poly_order, n_dim)
    return poly

def entries_count_for_no_cross_ndim_polynomial(poly_order, n_dim):
    n_entries = n_dim_no_cross_entries(poly_order, n_dim)
    poly = np.zeros([n_dim, n_entries])
    offset = 1
    for current_order in range(1, poly_order+1):
        for dim_i in range(n_dim):
            poly[dim_i, offset] = current_order
            offset += 1
    return poly

def n_dim_no_cross_entries(poly_order, n_dim):
    return 1 + n_dim * poly_order


def generate_full_three_dim_polynomial_powers(poly_order):
    _check_nonnegative_polynomial_power(poly_order)
    n_entries = entries_count_for_full_three_dim_polynomial(poly_order)
    poly = np.zeros([3, n_entries], dtype=int)
    offset = 1
    for current_order in range(1, poly_order + 1):
        for z_power in range(current_order + 1):
            for y_power in range(current_order + 1 - z_power):
                poly[0, offset] = current_order - y_power - z_power
                poly[1, offset] = y_power
                poly[2, offset] = z_power
                offset += 1
    return poly


def entries_count_for_full_three_dim_polynomial(poly_order):
    count = 0
    at_order = 1
    increment = 1
    for current_order in range(poly_order + 1):
        count += at_order
        increment += 1
        at_order += increment
    return count




def entries_count_for_full_two_dim_polynomial(N):
    return int((N + 1) * (N + 2) / 2)


def check_dimension(array, min_dimension_size):
    a_dim = array.ndim
    if a_dim < min_dimension_size:
        raise ArrayDimError(a_dim, min_dimension_size)
    
class ArrayDimError(RuntimeError):

    def __init__(self, array_dim, min_dim):
        message = f"\n  Required array dimenstion of size: {min_dim}\n  Supplied array has dimension {array_dim}"
        super().__init__(message)



class PolynomialCalculatorBase(ABC):

    def __init__(self, poly_max):
        self._poly_powers = self._calculate_powers(poly_max)
        self._n_dim = np.shape(self._poly_powers)[0]
        self._n_powers = np.shape(self._poly_powers)[1]

    def calculate(self, x):
        check_dimension(x,2)
        n_points = np.shape(x)[0]
        values = np.ones([n_points, self._n_powers])
        for dim in range(self._n_dim):
            for power_index, power in enumerate(self._poly_powers[dim, :]):
                values[:, power_index] = self._evaluate_polynomial(x, values, dim, power_index, power)
        return values

    def _evaluate_polynomial(self, x, values, dim, power_index, power):
        return np.multiply(values[:, power_index], np.power(x[:, dim], power))
    
    @abstractmethod
    def _calculate_powers(self, poly_max):
        return None




class GlobalNaivePolynomialCalculatorOneD(PolynomialCalculatorBase):

    def _calculate_powers(self, poly_max):
        return generate_full_one_dim_polynomial_powers(poly_max)
    


class LocalNaivePolynomialCalculatorOneD(PolynomialCalculatorBase):

    def _calculate_powers(self, poly_max):
        return generate_full_one_dim_polynomial_powers(poly_max)

    def _evaluate_polynomial(self, x, values, dim, power_index, power):
        x_lower = np.min(x[:, dim])
        x_upper = np.max(x[:, dim])
        x_range = x_upper - x_lower
        x_local = (x - x_lower) / x_range
        x_local = x_local.reshape((-1, 1))
        poly_val = np.multiply(values[:, power_index], np.power(x_local[:, dim], power))
        return poly_val

class LocalLegendrePolynomialCalculatorOneD(PolynomialCalculatorBase):

    def _calculate_powers(self, poly_max):
        return generate_full_one_dim_polynomial_powers(poly_max)

    def _evaluate_polynomial(self, x, values, dim, power_index, power):
        x_lower = np.min(x[:, dim])
        x_upper = np.max(x[:, dim])
        x_range = x_upper - x_lower
        x_z = (x - x_lower) / x_range
        x_local = 2 * x_z - 1
        x_local = x_local.reshape((-1, 1))
        poly_val = np.multiply(values[:, power_index], special.eval_legendre(power, x_local[:, dim]))
        return poly_val





class PolynomialCalculatorTwoD(PolynomialCalculatorBase):

    def _calculate_powers(self, poly_max):
        return generate_full_two_dim_polynomial_powers(poly_max)

class PolynomialCalculatorThreeD(PolynomialCalculatorBase):

  def _calculate_powers(self, poly_max):
    return generate_full_three_dim_polynomial_powers(poly_max)

class NoCrossPolynomialCalculatorTwoD(PolynomialCalculatorBase):

    def _calculate_powers(self, poly_max):
        return generate_nocross_two_dim_polynomial_powers(poly_max)
    
class ReducedPolynomialCaclulatorTwoD(PolynomialCalculatorBase):

    def _calculate_powers(self, poly_max):
        return generate_mid_and_edge_two_dim_polynimial_powers(poly_max)