from matcal.core.tests import MatcalUnitTest
from matcal.full_field.shapes import (ArrayDimError, GlobalNaivePolynomialCalculatorOneD, 
    LocalLegendrePolynomialCalculatorOneD, LocalNaivePolynomialCalculatorOneD, 
    NegativePolynomialOrderError, PolynomialCalculatorThreeD,
    generate_mid_and_edge_two_dim_polynimial_powers, generate_nocross_two_dim_polynomial_powers,
    entries_count_for_full_three_dim_polynomial, entries_count_for_full_two_dim_polynomial, 
    check_dimension, generate_full_one_dim_polynomial_powers,
    generate_full_three_dim_polynomial_powers, generate_full_two_dim_polynomial_powers,
    PolynomialCalculatorTwoD)
import numpy as np


        

class TestPolynomialPowerGenerators(MatcalUnitTest.MatcalUnitTest):

    def setUp(self):
        super().setUp(__file__)

    def confirm_poly_orders(self, order, goal, order_function):
        poly_orders = order_function(order)
        self.assert_close_arrays(poly_orders, goal)

    def test_full_one_d_zero(self):
        order = 0
        goal = [[0]]
        self.confirm_poly_orders(order, goal, generate_full_one_dim_polynomial_powers)
        
    def test_full_one_d_one(self):
        order = 1 
        goal = [[0, 1]]
        self.confirm_poly_orders(order, goal, generate_full_one_dim_polynomial_powers)

    def test_full_one_d_five(self):
        order = 5
        goal = [[0, 1,2, 3, 4, 5]]
        self.confirm_poly_orders(order, goal, generate_full_one_dim_polynomial_powers)

    def test_full_two_d_zero(self):
        order = 0 
        goal = [[0], [0]]
        self.confirm_poly_orders(order, goal, generate_full_two_dim_polynomial_powers)
    
    def test_full_two_d_one(self):
        order = 1
        goal = [[0, 1, 0], [0, 0, 1]]
        self.confirm_poly_orders(order, goal, generate_full_two_dim_polynomial_powers)

    def test_full_two_d_two(self):
        order = 2
        goal = [[0, 1, 0, 2, 1, 0],[0, 0, 1, 0, 1, 2]]
        self.confirm_poly_orders(order, goal, generate_full_two_dim_polynomial_powers)

    def test_full_two_d_four(self):
        order = 4
        goal = [[0, 1, 0, 2, 1, 0, 3, 2, 1, 0 , 4, 3, 2, 1, 0],[0, 0, 1, 0, 1, 2, 0 , 1, 2, 3,0, 1, 2, 3, 4]]
        self.confirm_poly_orders(order, goal, generate_full_two_dim_polynomial_powers)

    def test_no_cross_two_d_zero(self):
        order =0
        goal = [[0],[0]]
        self.confirm_poly_orders(order, goal, generate_nocross_two_dim_polynomial_powers)

    def test_no_cross_two_d_one(self):
        order =1
        goal = [[0, 1, 0],[0, 0, 1]]
        self.confirm_poly_orders(order, goal, generate_nocross_two_dim_polynomial_powers)

    def test_no_cross_two_d_twp(self):
        order =2
        goal = [[0, 1, 0, 2, 0],[0, 0, 1, 0, 2]]
        self.confirm_poly_orders(order, goal, generate_nocross_two_dim_polynomial_powers)

    def test_mid_and_edge_two_d_two(self):
        order =2
        goal = [[0, 1, 0, 2, 0, 1],[0, 0, 1, 0, 2, 1]]
        self.confirm_poly_orders(order, goal, generate_mid_and_edge_two_dim_polynimial_powers)

    def test_mid_and_edge_two_d_two(self):
        order =2
        goal = [[0, 1, 0, 2, 0, 1],[0, 0, 1, 0, 2, 1]]
        self.confirm_poly_orders(order, goal, generate_mid_and_edge_two_dim_polynimial_powers)

    def test_mid_and_edge_two_d_three(self):
        order = 3
        goal = [[0, 1, 0, 2, 0, 1, 3, 0],[0, 0, 1, 0, 2, 1, 0, 3]]
        self.confirm_poly_orders(order, goal, generate_mid_and_edge_two_dim_polynimial_powers)

    def test_mid_and_edge_two_d_four(self):
        order = 4
        goal = [[0, 1, 0, 2, 0, 1, 3, 0, 4, 0, 2],[0, 0, 1, 0, 2, 1, 0, 3, 0, 4, 2]]
        self.confirm_poly_orders(order, goal, generate_mid_and_edge_two_dim_polynimial_powers)

    def test_full_three_d_zero(self):
        order = 0 
        goal = [[0], [0], [0]]
        self.confirm_poly_orders(order, goal, generate_full_three_dim_polynomial_powers)

    def test_full_three_d_one(self):
        order = 1 
        goal = [[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        self.confirm_poly_orders(order, goal, generate_full_three_dim_polynomial_powers)

    def test_full_three_d_two(self):
        order = 2 
        goal = [[0, 1, 0, 0, 2, 1, 0, 1, 0, 0], [0, 0, 1, 0, 0, 1, 2, 0, 1, 0 ], [0, 0, 0, 1, 0, 0, 0, 1, 1, 2]]
        self.confirm_poly_orders(order, goal, generate_full_three_dim_polynomial_powers)

    def test_sum_decreasing_series(self):
        self.assertEqual(entries_count_for_full_two_dim_polynomial(0), 1)
        self.assertEqual(entries_count_for_full_two_dim_polynomial(1), 3)
        self.assertEqual(entries_count_for_full_two_dim_polynomial(2), 6)
        self.assertEqual(entries_count_for_full_two_dim_polynomial(4), 15)

    def test_3d_permutation_count(self):
        self.assertEqual(entries_count_for_full_three_dim_polynomial(0), 1)
        self.assertEqual(entries_count_for_full_three_dim_polynomial(1), 4)
        self.assertEqual(entries_count_for_full_three_dim_polynomial(2) , 10)
        self.assertEqual(entries_count_for_full_three_dim_polynomial(3) , 20)
        self.assertEqual(entries_count_for_full_three_dim_polynomial(4) , 35)

    def test_raise_error_with_negative_powers(self):
        with self.assertRaises(NegativePolynomialOrderError):
            generate_full_one_dim_polynomial_powers(-1)
        with self.assertRaises(NegativePolynomialOrderError):
            generate_full_two_dim_polynomial_powers(-1)
        with self.assertRaises(NegativePolynomialOrderError):
            generate_full_three_dim_polynomial_powers(-1)
            
class Test_check_dimension(MatcalUnitTest.MatcalUnitTest):

    def setUp(self):
        super().setUp(__file__)

    def test_pass(self):
        n_entry_per_dim = 4
        array_size = []
        for n_dim in range(1, 10):
            array_size.append(n_entry_per_dim)
            test_array = np.zeros(array_size)
            for test_dim in range(n_dim):
                check_dimension(test_array, test_dim)

    def test_fail(self):
        n_entry_per_dim = 4
        array_size = []
        max_dim = 10
        for n_dim in range(1, max_dim):
            array_size.append(n_entry_per_dim)
            test_array = np.zeros(array_size)
            for test_dim in range(n_dim+1, max_dim-n_dim):
                with self.assertRaises(ArrayDimError):
                    check_dimension(test_array, test_dim)


class TestPolynomialCalculatorsOneD(MatcalUnitTest.MatcalUnitTest):

    def setUp(self):
        super().setUp(__file__)
        self.pc_classes = [GlobalNaivePolynomialCalculatorOneD, LocalNaivePolynomialCalculatorOneD, LocalLegendrePolynomialCalculatorOneD]

    def confirm_calculate_0_1_3(self, order, goal, pc_class):
        pc = pc_class(order)
        x = np.array([0, 1, 3]).reshape([-1, 1])
        self.assert_close_arrays(pc.calculate(x), goal)

    def confirm_calculate_n1_1_3_4(self, order, goal, pc_class):
        pc = pc_class(order)
        x = np.array([-1, 1, 3, 4]).reshape([-1, 1])
        self.assert_close_arrays(pc.calculate(x), goal)

    def test_const_global(self):
        order = 0
        goal = np.array([[1], [1], [1]])
        self.confirm_calculate_0_1_3(order, goal, GlobalNaivePolynomialCalculatorOneD)

    def test_linear_global(self):
        order = 1
        goal = np.array([[1, 0], [1, 1], [1, 3]])
        self.confirm_calculate_0_1_3(order, goal, GlobalNaivePolynomialCalculatorOneD)

    def test_quad_global(self):
        order = 2
        goal = np.array([[1, 0, 0], [1, 1, 1], [1, 3, 9]])
        self.confirm_calculate_0_1_3(order, goal, GlobalNaivePolynomialCalculatorOneD)

    def test_quad_global2(self):
        order = 2
        goal = np.array([[1, -1, 1], [1, 1, 1], [1, 3, 9], [1, 4, 16]])
        self.confirm_calculate_n1_1_3_4(order, goal, GlobalNaivePolynomialCalculatorOneD)

    def test_bad_array_format_raise_error(self):
        order = 2
        x = np.array([0,1,2])
        for pc_class in self.pc_classes:
            pc = pc_class(order)
            with self.assertRaises(ArrayDimError):
                pc.calculate(x)

    def test_const_local(self):
        order = 0
        goal = [[1], [1], [1]]
        self.confirm_calculate_0_1_3(order, goal, LocalNaivePolynomialCalculatorOneD)

    def test_const_local2(self):
        order = 0
        goal = [[1], [1], [1], [1]]
        self.confirm_calculate_n1_1_3_4(order, goal, LocalNaivePolynomialCalculatorOneD)

    def test_lin_local(self):
        order = 1
        goal = [[1,0], [1, 1/3], [1, 1]]
        self.confirm_calculate_0_1_3(order, goal, LocalNaivePolynomialCalculatorOneD)

    def test_lin_local2(self):
        order = 1
        goal = [[1, 0], [1, 2/5], [1, 4/5], [1, 1]]
        self.confirm_calculate_n1_1_3_4(order, goal, LocalNaivePolynomialCalculatorOneD)

    def test_quad_local(self):
        order = 2
        goal = [[1, 0, 0], [1, 1/3, 1/9], [1, 1, 1]]
        self.confirm_calculate_0_1_3(order, goal, LocalNaivePolynomialCalculatorOneD)

    def test_quad_local2(self):
        order = 2
        goal = [[1, 0, 0], [1, 2/5, 4/25], [1, 4/5, 16/25], [1, 1, 1]]
        self.confirm_calculate_n1_1_3_4(order, goal, LocalNaivePolynomialCalculatorOneD)

    def test_const_legendre(self):
        order = 0
        goal = [[1], [1], [1]]
        self.confirm_calculate_0_1_3(order, goal, LocalLegendrePolynomialCalculatorOneD)

    def test_quad_legendre(self):
        order = 2
        goal = np.array([[1, -1, 1], [1, -1/3, -1/3], [1, 1, 1]])
        self.confirm_calculate_0_1_3(order, goal, LocalLegendrePolynomialCalculatorOneD)

class TestPolynomialCalculatorsTwoD(MatcalUnitTest.MatcalUnitTest):

    def setUp(self):
        super().setUp(__file__)

    def confirm_calculate(self, order, goal, poly_class):
        pc = poly_class(order)
        points = np.array([[0,0],[1,2],[-2.5, 4], [-3, -2]])
        poly_vals = pc.calculate(points)
        self.assert_close_arrays(poly_vals, goal)

    def test_constant(self):
        order = 0 
        goal = [[1],[1],[1], [1]]
        self.confirm_calculate(order, goal, PolynomialCalculatorTwoD)
    
    def test_linear(self):
        order = 1
        goal = [[1, 0, 0],[1, 1, 2],[1, -2.5, 4], [1, -3, -2]]
        self.confirm_calculate(order, goal, PolynomialCalculatorTwoD)

    def test_quad(self):
        order = 2
        goal = [[1, 0, 0, 0, 0, 0],[1, 1, 2, 1, 2, 4],[1, -2.5, 4, 2.5**2, -10, 16], [1, -3, -2, 9, 6, 4]]
        self.confirm_calculate(order, goal, PolynomialCalculatorTwoD)

class TestPolynomialCalculatorThreeD(MatcalUnitTest.MatcalUnitTest):

    def setUp(self):
        super().setUp(__file__)

    def confirm_calculate(self, order, goal, poly_class):
        pc = poly_class(order)
        points = np.array([[0,0,0],[1,2,3],[-2.5, 4, 1], [-3, -2, -5]])
        poly_vals = pc.calculate(points)
        self.assert_close_arrays(poly_vals, goal)
    
    def test_constant(self):
        order = 0 
        goal = [[1],[1],[1], [1]]
        self.confirm_calculate(order, goal, PolynomialCalculatorThreeD)
    
    def test_linear(self):
        order = 1
        goal = [[1, 0, 0, 0],[1, 1, 2, 3],[1, -2.5, 4, 1], [1, -3, -2, -5]]
        self.confirm_calculate(order, goal, PolynomialCalculatorThreeD)

    def test_quad(self):
        order = 2
        goal = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],[1, 1, 2, 3, 1, 2, 4, 3, 6, 9],[1, -2.5, 4, 1, 2.5**2, -10, 16, -2.5, 4, 1], [1, -3, -2, -5, 9, 6, 4, 15, 10, 25]])
        self.confirm_calculate(order, goal, PolynomialCalculatorThreeD)

