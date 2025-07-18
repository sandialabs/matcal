from matcal.core.tests import MatcalUnitTest
import numpy as np

from matcal.full_field.hwd import shapes
from matcal.full_field.hwd import LinearConditioner, NoCrossTwoDPolynomialHWD, OneDPolynomialHWD, PolynomialMomentMatrixGeneratorOneD, ReducedTwoDPolynomialHWD, ReducedTwoDPolynomialROHWD, TwoDPolynomialHWD


class TestOneDimPolynomialMomentMatrixGenerator(MatcalUnitTest.MatcalUnitTest):

    def setUp(self):
        super().setUp(__file__)

    MatGen = PolynomialMomentMatrixGeneratorOneD

    class MatGenSpy(MatGen):
        
        def spy_max_depth(self):
            return self._max_tree_depth
        
        def spy_use_scaling(self):
            return self._use_scaling
        
        def spy_min_cluster_size(self):
            return self._min_cluster_size

        def spy_wavelet_per_domain(self):
            return self._wavelets_per_domain

    def test_confirm_default_init(self):
        poly_order = 4
        gen = self.MatGenSpy(poly_order)
        self.assertEqual(gen.spy_max_depth(), 5)
        self.assertTrue(gen.spy_use_scaling())
        self.assertEqual(gen.spy_min_cluster_size(), poly_order+1)
        self.assertEqual(gen.spy_wavelet_per_domain(), poly_order+1)
        self.assertEqual(gen._active_side_index, 0)
        self.assertFalse(gen.is_built)

    def test_confirm_non_default_init(self):
        poly_order = 2
        depth = 3
        gen = self.MatGenSpy(poly_order, depth, False)
        self.assertEqual(gen.spy_max_depth(), depth)
        self.assertFalse(gen.spy_use_scaling())
        self.assertEqual(gen.spy_min_cluster_size(), poly_order+1)
        self.assertEqual(gen.spy_wavelet_per_domain(), poly_order+1)
        self.assertEqual(gen._active_side_index, 0)


    def test_raise_error_if_poly_order_non_posiitive_int(self):
        with self.assertRaises(shapes.NegativePolynomialOrderError):
            gen = self.MatGenSpy(-1)
        with self.assertRaises(self.MatGenSpy.NonIntegerPolynomialOrderError):
            gen = self.MatGenSpy(1.4)

    def test_raise_error_if_max_depth_is_non_counting_number(self):
        poly_order = 1
        bad_depths = [-1, -10, 1.5, 1.0]
        for bd in bad_depths:
            with self.assertRaises(self.MatGenSpy.NonCountingNumberTreeDepthError):
                self.MatGenSpy(poly_order, max_tree_depth=bd)

    def test_raise_error_if_scaling_flag_non_bool(self):
        poly_order = 0
        bad_flags = [1, -4, .44, 'a', 'true', 0]
        for bf in bad_flags:
            with self.assertRaises(self.MatGenSpy.NonBoolianScaleInputsFlagError):
                self.MatGenSpy(poly_order, scale_inputs=bf)

    def test_build_flag_is_true_after_building(self):
        poly_order = 0
        max_depth = 0
        use_scaling = False
        locations = np.array([0,1,2,3,4,5]).reshape(-1, 1)
        generator = self.MatGenSpy(poly_order, max_depth, use_scaling)
        M = generator.build_and_populate(locations)
        self.assertTrue(generator.is_built)
        

    def test_build_and_populate_zero_order_zero_depth(self):
        poly_order = 0
        max_depth = 0
        use_scaling = False
        locations = np.array([0,1,2,3,4,5]).reshape(-1, 1)
        goal = np.ones([len(locations), 1])
        self._check_build_and_populate(poly_order, max_depth, use_scaling, locations, goal)

    def test_build_and_populate_first_order_zero_depth(self):
        poly_order = 1
        max_depth = 0
        use_scaling = False
        locations = np.array([0,1,2,3,4,5]).reshape(-1, 1)
        goal = np.ones([len(locations), poly_order+1])
        goal[:, 1] = locations.flatten()
        self._check_build_and_populate(poly_order, max_depth, use_scaling, locations, goal)

    def _check_build_and_populate(self, poly_order, max_depth, use_scaling, locations, goal):
        generator = self.MatGenSpy(poly_order, max_depth, use_scaling)
        M = generator.build_and_populate(locations)
        self.assert_close_arrays(goal, M)

    def test_build_and_populate_second_order_zero_depth(self):
        poly_order = 2
        max_depth = 0
        use_scaling = False
        locations = np.array([0,1,2,3,4,5]).reshape(-1, 1)
        goal = np.ones([len(locations), poly_order+1])
        goal[:, 1] = locations.flatten()
        goal[:, 2] = np.power(locations, 2).flatten()
        self._check_build_and_populate(poly_order, max_depth, use_scaling, locations, goal)

    def test_build_and_populate_zero_order_one_depth(self):
        poly_order = 0
        max_depth = 1
        use_scaling = False
        locations = np.array([0,1,2,3,4,5]).reshape(-1, 1)
        generator = self.MatGenSpy(poly_order, max_depth, use_scaling)
        M = generator.build_and_populate(locations)
        goal = np.ones([len(locations), (max_depth+1)*(poly_order+1)])
        if np.abs(M[0,1]) < 1e-8:
            zero_idxs = [0,1,2]
        else:
            zero_idxs = [3,4,5]
        goal[zero_idxs, 1] = 0
        self.assert_close_arrays(goal, M)

    def test_build_and_populate_first_order_one_depth(self):
        poly_order = 1
        max_depth = 1
        use_scaling = False
        locations = np.array([0,1,2,3,4,5]).reshape(-1, 1)
        generator = self.MatGenSpy(poly_order, max_depth, use_scaling)
        M = generator.build_and_populate(locations)
        goal = np.ones([len(locations), (max_depth+1)*(poly_order+1)])
        if np.abs(M[0,2]) < 1e-8:
            zero_idxs = [0,1,2]
        else:
            zero_idxs = [3,4,5]
        goal[:,1] = locations.flatten()
        goal[:,3] = locations.flatten()
        goal[zero_idxs, 2:] = 0
        self.assert_close_arrays(goal, M)

    def test_populate_first_order_one_depth(self):
        poly_order = 1
        max_depth = 1
        use_scaling = False
        locations = np.array([0,1,2,3,4,5]).reshape(-1, 1)
        locations2 = np.array([-5, -2, 1, 2, 5, 7, 9])
        generator = self.MatGenSpy(poly_order, max_depth, use_scaling)
        generator.build_and_populate(locations)
        M = generator.populate(locations2.reshape(-1,1))
        goal = np.ones([len(locations2), (max_depth+1)*(poly_order+1)])
        if np.abs(M[0,2]) < 1e-8:
            zero_idxs = [0,1,2,3]
        else:
            zero_idxs = [4,5,6]
        goal[:,1] = locations2.flatten()
        goal[:,3] = locations2.flatten()
        goal[zero_idxs, 2:] = 0
        self.assert_close_arrays(goal, M)

    def test_populate_first_order_two_depth(self):
        np.random.seed(0)
        poly_order = 1
        max_depth = 2
        use_scaling = False
        locations = np.array([-20,-19,-15,-14,14,15, 19, 20]).reshape(-1, 1)
        locations2 = np.array([-19.5, -14.5, 14.5, 19.5, -19.5, -14.5, 14.5, 19.5])
        generator = self.MatGenSpy(poly_order, max_depth, use_scaling)
        generator.build_and_populate(locations)
        M = generator.populate(locations2.reshape(-1,1))
        goal = np.array(
               [[  1.,  -19.5,   1.,  -19.5,   1.,  -19.5,   0.,    0. ],
                [  1.,  -14.5,   1.,  -14.5,   0.,    0. ,   0.,    0. ],
                [  1.,   14.5,   0.,    0. ,   0.,    0. ,   1.,   14.5],
                [  1.,   19.5,   0.,    0. ,   0.,    0. ,   0.,    0. ],
                [  1.,  -19.5,   1.,  -19.5,   1.,  -19.5,   0.,    0. ],
                [  1.,  -14.5,   1.,  -14.5,   0.,    0. ,   0.,    0. ],
                [  1.,   14.5,   0.,    0. ,   0.,    0. ,   1.,   14.5],
                [  1.,   19.5,   0.,    0. ,   0.,    0. ,   0.,    0. ]]
                )

        self.assertTrue(np.allclose(goal, M))


    def test_populate_first_order_one_depth_with_scaling(self):
        poly_order = 1
        max_depth = 1
        locations = np.array([0,1,2,3,4,5]).reshape(-1, 1)
        def cond(x):
            return (2/5) * (x-0) - 1
        lc = np.array([-1,-3/5, -1/5, 1/5, 3/5, 1])
        locations2 = np.array([-5, -2, 1, 2, 5, 7, 9])
        lc2 = cond(locations2)
        generator = self.MatGenSpy(poly_order, max_depth)
        generator.build_and_populate(locations)
        M = generator.populate(locations2.reshape(-1,1))
        goal = np.ones([len(locations2), (max_depth+1)*(poly_order+1)])
        if np.abs(M[0,2]) < 1e-8:
            zero_idxs = [0,1,2,3]
        else:
            zero_idxs = [4,5,6]
        goal[:,1] = lc2.flatten()
        goal[:,3] = lc2.flatten()
        goal[zero_idxs, 2:] = 0
        self.assert_close_arrays(goal, M)

class TestConditioning(MatcalUnitTest.MatcalUnitTest):

        def setUp(self):
            super().setUp(__file__)

        def test_scale_1_to_10_to_n1_to_1(self):
            n_pts = 5
            x = np.linspace(1, 10, n_pts).reshape(-1, 1)
            goal = np.linspace(-1, 1, n_pts).reshape(-1,1)
            scale_tool = LinearConditioner()
            x_c = scale_tool.fit_and_condition(x)
            self.assert_close_arrays(x_c, goal)
        
        def test_condition_on_already_conditioned_tools(self):
            n_pts = 5
            x = np.linspace(1, 10, n_pts).reshape(-1, 1)
            x2 = np.linspace(1, 10, n_pts*2).reshape(-1, 1)
            goal = np.linspace(-1, 1, n_pts*2).reshape(-1,1)
            scale_tool = LinearConditioner()
            scale_tool.fit_and_condition(x)
            x_c = scale_tool.condition(x2)
            self.assert_close_arrays(x_c, goal)

        def test_condition_on_already_conditioned_tools_diff_range(self):
            n_pts = 5
            x = np.linspace(1, 10, n_pts).reshape(-1, 1)
            x2 = np.linspace(1, 5.5, n_pts*2).reshape(-1, 1)
            goal = np.linspace(-1, 0, n_pts*2).reshape(-1,1)
            scale_tool = LinearConditioner()
            scale_tool.fit_and_condition(x)
            x_c = scale_tool.condition(x2)
            self.assert_close_arrays(x_c, goal)

        def test_multi_dimensional(self):
            n_pts = 5
            x = np.linspace(0, 4, n_pts)
            y = np.linspace(-1, 10, n_pts)
            z = np.linspace(2, 4, n_pts)
            p = np.array([x,y,z]).T 
            goal_line = np.linspace(-1, 1, n_pts)
            goal = np.array([goal_line, goal_line, goal_line]).T
            scale_tool = LinearConditioner()
            p_c = scale_tool.fit_and_condition(p)
            self.assert_close_arrays(p_c, goal)


class TestOneDPolynomialHWD(MatcalUnitTest.MatcalUnitTest):

    def setUp(self):
        super().setUp(__file__)

    def poly(self, x, *cs):
        value = np.zeros_like(x)
        for p, c in enumerate(cs):
            value += c * np.power(x, p)
        return value

    def test_simple_polynomial_map(self):
        n_pts = 21
        x = np.linspace(0, 10, n_pts)
        x2 = np.linspace(0, 10, n_pts+3)
        cs = [1, 1]
        y = self.poly(x, *cs)
        y2 = self.poly(x2, *cs)
        max_depth = 2
        poly_oder = 3

        hwd = OneDPolynomialHWD(poly_oder, max_depth)
        a = hwd.map_data(y, x.reshape(-1, 1))
        Q_ref, R_ref = hwd.get_basis()
        a2_mapped = hwd.map_data(y2, x2.reshape(-1, 1), R_ref)
        self.assert_close_arrays(a, a2_mapped)

    def lin_cos(self, x, c_cos, f_cos, c_lin):
        value = c_lin * x
        value += c_cos * np.cos(2*np.pi * f_cos * x)
        return value    
    
    def test_lin_cos_map(self):
        n_pts = 40
        x = np.linspace(0, 2, n_pts)
        x2 = np.linspace(0, 2, n_pts+4)
        cs = [1, 1/4, 2]
        y = self.lin_cos(x, *cs)
        y2 = self.lin_cos(x2, *cs)
        max_depth = 2
        poly_oder = 7
        
        hwd = OneDPolynomialHWD(poly_oder, max_depth, return_m=True)
        a = hwd.map_data(y, x.reshape(-1, 1))
        Q_ref, R_ref, M_ref = hwd.get_basis()


        a2_mapped = hwd.map_data(y2, x2.reshape(-1, 1), R_ref)
        sig_args = np.argwhere(np.abs(a) > np.max(np.abs(a))*1e-3)
        delta = a - a2_mapped

        
        self.assert_close_arrays(a, a2_mapped)

    def test_fail_when_n_points_less_than_wavelets(self):
        n_pts = 6
        poly_order = 10
        depth = 2
        hwd = OneDPolynomialHWD(poly_order, depth)
        x = np.linspace(0, 1, n_pts)
        y = np.linspace(10,20, n_pts)
        with self.assertRaises(OneDPolynomialHWD.ExcessiveWaveletsError):
            hwd.map_data(y, x.reshape(-1, 1))    

class TestTwoDPolynomialHWD(MatcalUnitTest.MatcalUnitTest):

    def setUp(self):
        super().setUp(__file__)

    def lin_cos(self, x, y, a_cos, f_cos, c_lin):
        value =  c_lin * (x + y)
        value += a_cos * np.cos(2*np.pi * (y))
        value += a_cos * np.cos(2*np.pi * (x))
        return value
    
    def test_close_map(self):
        max_depth = 2
        max_poly = 12
        n_pts = (max_poly+1)**2 * (2**max_depth) + 10
        add_pts = 10
        x_ref = np.linspace(0, 2, n_pts)
        y_ref = np.linspace(-1, 1, n_pts)
        x_ref, y_ref = np.meshgrid(x_ref, y_ref)
        x_ref = x_ref.flatten()
        y_ref = y_ref.flatten()
        loc_ref = np.array([x_ref, y_ref]).T
        x = np.linspace(0, 2, n_pts+add_pts)
        y = np.linspace(-1, 1, n_pts+add_pts)
        x, y = np.meshgrid(x, y)
        x = x.flatten()
        y = y.flatten()
        loc = np.array([x,y]).T
        
        consts = [.5, 1, 1]
        consts = [2, 1, 1]
        z_ref = self.lin_cos(x_ref,y_ref, *consts)
        z = self.lin_cos(x,y, *consts)

        hwd = TwoDPolynomialHWD(max_poly, max_depth, return_m=True)
        a_ref = hwd.map_data(z_ref, loc_ref)
        Q_ref, R_ref, M_ref = hwd.get_basis()
        a = hwd.map_data(z, loc)
        Q, R, M = hwd.get_basis()
        a_map_me = np.dot(R_ref, np.linalg.solve(R, a))
        a_map = hwd.map_data(z, R_ref=R_ref)
        

        weight_error = np.linalg.norm(a_ref-a_map)
        self.assertTrue(weight_error < 1e-6)

class TestNotFullTwoDPolynomialHWD(MatcalUnitTest.MatcalUnitTest):

    def setUp(self):
        super().setUp(__file__)

    def lin_cos(self, x, y, a_cos, f_cos, c_lin):
        value =  c_lin * (x + y)
        value += a_cos * np.cos(2*np.pi * (x))
        value += a_cos * np.cos(2*np.pi * (y))
        return value
    
    def exp(self, x, y):
        value =  np.exp((y))
        value += np.exp(x)
        return value

    def test_close_lin_cos(self):
        max_depth = 3
        max_poly = 12
        add_pts = 10
        n_pts = 3*(max_poly+1) * (max_depth+1) + add_pts
        x_ref = np.linspace(0, 2, n_pts)
        y_ref = np.linspace(-1, 1, n_pts)
        x_ref, y_ref = np.meshgrid(x_ref, y_ref)
        x_ref = x_ref.flatten()
        y_ref = y_ref.flatten()
        loc_ref = np.array([x_ref, y_ref]).T
        x = np.linspace(0, 2, n_pts+add_pts)
        y = np.linspace(-1, 1, n_pts+add_pts)
        x, y = np.meshgrid(x, y)
        x = x.flatten()
        y = y.flatten()
        loc = np.array([x,y]).T
        
        consts = [.5, 1, 1]
        consts = [2, 1, 1]
        z_ref = self.lin_cos(x_ref,y_ref, *consts)
        z = self.lin_cos(x,y, *consts)

        hwd = ReducedTwoDPolynomialHWD(max_poly, max_depth)
        a_ref = hwd.map_data(z_ref, loc_ref)
        Q_ref, R_ref = hwd.get_basis()
        a = hwd.map_data(z, loc)
        Q, R = hwd.get_basis()
        a_map = hwd.map_data(z, R_ref=R_ref)
        

        weight_err = np.linalg.norm(a_ref-a_map)
        self.assertTrue(weight_err < 1e-6)

    def test_close_map_exp(self):
        max_depth = 0
        max_poly = 10
        add_pts = 10
        n_pts = 3*(max_poly+1) * (max_depth+1) + add_pts
        x_ref = np.linspace(0, 2, n_pts)
        y_ref = np.linspace(-1, 1, n_pts)
        x_ref, y_ref = np.meshgrid(x_ref, y_ref)
        x_ref = x_ref.flatten()
        y_ref = y_ref.flatten()
        loc_ref = np.array([x_ref, y_ref]).T
        x = np.linspace(0, 2, n_pts+add_pts)
        y = np.linspace(-1, 1, n_pts+add_pts)
        x, y = np.meshgrid(x, y)
        x = x.flatten()
        y = y.flatten()
        loc = np.array([x,y]).T
        
        z_ref = self.exp(x_ref,y_ref)
        z = self.exp(x,y)

        hwd = ReducedTwoDPolynomialHWD(max_poly, max_depth)
        a_ref = hwd.map_data(z_ref, loc_ref)
        Q_ref, R_ref = hwd.get_basis()
        a = hwd.map_data(z, loc)
        Q, R = hwd.get_basis()
        a_map = hwd.map_data(z, R_ref=R_ref)
        

        weight_err = np.linalg.norm(a_ref-a_map)
        self.assertTrue(weight_err < 1e-6)

    def poly(self, x,y):
        return np.power(x+y, 2)


    def test_close_map_poly(self):
        max_depth = 0
        max_poly = 2
        add_pts = 10
        n_pts = 3*(max_poly+1) * (max_depth+1) + add_pts
        x_ref = np.linspace(0, 2, n_pts)
        y_ref = np.linspace(-1, 1, n_pts)
        x_ref, y_ref = np.meshgrid(x_ref, y_ref)
        x_ref = x_ref.flatten()
        y_ref = y_ref.flatten()
        loc_ref = np.array([x_ref, y_ref]).T
        x = np.linspace(0, 2, n_pts+add_pts)
        y = np.linspace(-1, 1, n_pts+add_pts)
        x, y = np.meshgrid(x, y)
        x = x.flatten()
        y = y.flatten()
        loc = np.array([x,y]).T
        
        z_ref = self.poly(x_ref,y_ref)
        z = self.poly(x,y)

        hwd = ReducedTwoDPolynomialHWD(max_poly, max_depth)
        a_ref = hwd.map_data(z_ref, loc_ref)
        Q_ref, R_ref = hwd.get_basis()
        a = hwd.map_data(z, loc)
        Q, R = hwd.get_basis()
        a_map = hwd.map_data(z, R_ref=R_ref)
        

        weight_err = np.linalg.norm(a_ref-a_map)
        self.assertTrue(weight_err < 1e-6)


class TestReducedOrderHWD(MatcalUnitTest.MatcalUnitTest):

    def setUp(self):
        super().setUp(__file__)

    def test_only_const_lin(self):
        max_depth = 0
        max_poly = 5
        n_pts = 25
        n_dim = 2
    
        def test_fun(points):
            return 1 + points[:,0] + 2 * points[:,1]

        points = np.random.uniform(-1, 1, [n_pts, n_dim])
        values = test_fun(points)

        ro = ReducedTwoDPolynomialROHWD(max_poly, max_depth)
        ro.build_compressed_space(points, values)
        c = ro.map_data(values)
        self.assertEqual(len(c), 3)
        self.assert_close_arrays(values, ro._Q.dot(c))
        
    def test_only_quad(self):
        max_depth = 0
        max_poly = 5
        n_pts = 25
        n_dim = 2
    
        def test_fun(points):
            return 1 + points[:,0] + np.power(points[:,0], 2)

        points = np.random.uniform(-1, 1, [n_pts, n_dim])
        inc_x_sort = np.argsort(points[:,0])
        points = points[inc_x_sort, :]
        values = test_fun(points)

        ro = ReducedTwoDPolynomialROHWD(max_poly, max_depth, 1e-4)
        ro.build_compressed_space(points, values)
        c = ro.map_data(values)
        self.assert_close_arrays(values, ro._Q.dot(c))
        
        
    
    
