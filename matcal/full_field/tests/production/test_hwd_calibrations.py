
from matcal import *
from matcal.core.tests.MatcalUnitTest import MatcalUnitTest

def gauss_decay_fuction(**params):
    import numpy as np
    n_pts= params['num_axial_points']
    n_steps = params['num_time_steps']
    x = np.linspace(0, 1, n_pts)
    y = np.linspace(0, 1, n_pts)
    X, Y = np.meshgrid(x, y)
    x = X.flatten()
    y = Y.flatten()
    loc = np.array([x, y]).T
    T = np.zeros([n_pts**2, n_steps])
    time = list(range(n_steps))
    def gauss(loc, L_char):
        center = np.ones([1,2]) * .5
        radius = np.linalg.norm(loc - center, axis=1)
        scaled_rad = radius / L_char
        T = np.exp(-np.power(scaled_rad, 2))
        return T
    
    T[:,0] = gauss(loc, params['L'])
    for i in range(1, n_steps):
        T[:, i] = .5 * T[:,i-1]
    return {'time':time, 'x':x, 'y':y, 'T':T.T}


class OneParameterPythonWaveletCalibrationTest(MatcalUnitTest):

    def setUp(self):
        super().setUp(__file__)
        self._error_tol = 1.0e-3

    def test_characteristic_length_calibration_gradient(self):
        L_goal = .1
        test_params = {'num_axial_points':20, "num_time_steps":5, "L":L_goal}
        goal_results = gauss_decay_fuction(**test_params)


        field_data = convert_dictionary_to_field_data(goal_results, ['x', 'y'])

        model = PythonModel(gauss_decay_fuction, field_coordinates=['x', 'y'])
        model.add_constants(num_axial_points = test_params['num_axial_points'], num_time_steps = test_params['num_time_steps'])

        poly_order = 8
        max_depth = 1
        my_objective = PolynomialHWDObjective(None, "T", max_depth=max_depth, polynomial_order=poly_order)

        L = Parameter("L", 1e-4, 1)
        calibration = GradientCalibrationStudy(L)
        calibration.add_evaluation_set(model, my_objective, field_data)
        calibration.set_core_limit(8)
        

        results = calibration.launch()
        L_cal = results.outcome['best:L']

        self.assertAlmostEqual(L_cal, L_goal, delta = self._error_tol * L_goal)

    def test_characteristic_length_calibration_pattern_search(self):
        L_goal = .1
        test_params = {'num_axial_points':20, "num_time_steps":5, "L":L_goal}
        goal_results = gauss_decay_fuction(**test_params)


        field_data = convert_dictionary_to_field_data(goal_results, ['x', 'y'])

        model = PythonModel(gauss_decay_fuction, field_coordinates=['x', 'y'])
        model.add_constants(num_axial_points = test_params['num_axial_points'], num_time_steps = test_params['num_time_steps'])

        poly_order = 3
        max_depth = 3
        my_objective = PolynomialHWDObjective(None, "T", max_depth=max_depth, polynomial_order=poly_order)

        L = Parameter("L", 1e-4, 1)

        calibration = PatternSearchCalibrationStudy(L)
        calibration.add_evaluation_set(model, my_objective, field_data)
        calibration.set_core_limit(8)
        

        results = calibration.launch()
        L_cal = results.outcome['best:L']

        self._error_tol = 1.0e-2
        self.assertAlmostEqual(L_cal, L_goal, delta = self._error_tol * L_goal)

def cos_wiggle_fuction(**params):
    import numpy as np
    n_pts= params['num_axial_points']
    n_steps = params['num_time_steps']
    x = np.linspace(0, 1, n_pts)
    y = np.linspace(0, 1, n_pts)
    X, Y = np.meshgrid(x, y)
    x = X.flatten()
    y = Y.flatten()
    loc = np.array([x, y]).T
    T = np.zeros([n_pts**2, n_steps])
    time = list(range(n_steps))
    def wiggle(loc, time, L_char, T_char):
        T = np.cos(np.pi * (loc[:,0] / L_char + time /T_char))
        return T
    
    
    for i in range(n_steps):
        T[:, i] = wiggle(loc, i, params['Lc'], params['tc'])
    return {'time':time, 'x':x, 'y':y, 'T':T.T}

def cos_double_wiggle_fuction(**params):
    import numpy as np
    n_pts= params['num_axial_points']
    n_steps = params['num_time_steps']
    x = np.linspace(0, 1, n_pts)
    y = np.linspace(0, 1, n_pts)
    X, Y = np.meshgrid(x, y)
    x = X.flatten()
    y = Y.flatten()
    loc = np.array([x, y]).T
    T = np.zeros([n_pts**2, n_steps])
    time = list(range(n_steps))
    def wiggle(loc, time, L_char, T_char):
        T = np.cos(np.pi * (loc[:,0] / L_char + time /T_char)) 
        T += np.cos(np.pi * (loc[:,1] / L_char + time /T_char))
        return T
    
    
    for i in range(n_steps):
        T[:, i] = wiggle(loc, i, params['Lc'], params['tc'])
    return {'time':time, 'x':x, 'y':y, 'T':T.T}

class TwoParameterPythonWaveletCalibrationTest(MatcalUnitTest):

    def setUp(self):
        super().setUp(__file__)
        self._error_tol = 1.0e-3

    def test_characteristic_length_and_time_gradient_calibration(self):
        Lc_goal = .5
        tc_goal = 5
        test_params = {'num_axial_points':20, "num_time_steps":5, "Lc":Lc_goal, 'tc':tc_goal}
        goal_results = cos_wiggle_fuction(**test_params)


        field_data = convert_dictionary_to_field_data(goal_results, ['x', 'y'])

        model = PythonModel(cos_wiggle_fuction, field_coordinates=['x', 'y'])
        model.add_constants(num_axial_points = test_params['num_axial_points'], num_time_steps = test_params['num_time_steps'])

        poly_order = 4
        max_depth = 4
        my_objective = PolynomialHWDObjective(None, "T", max_depth=max_depth, polynomial_order=poly_order)

        Lc = Parameter("Lc", .1, 1)
        tc = Parameter('tc', 1, 10)
        calibration = GradientCalibrationStudy(Lc, tc)

        calibration.add_evaluation_set(model, my_objective, field_data)
        calibration.set_core_limit(8)
        

        results = calibration.launch()
        L_cal = results.outcome['best:Lc']
        tc_cal = results.outcome['best:tc']

        self.assertAlmostEqual(L_cal, Lc_goal, delta = self._error_tol * Lc_goal)
        self.assertAlmostEqual(tc_cal, tc_goal, delta = self._error_tol * tc_cal)

    def test_characteristic_length_and_time_gradient_calibration(self):
        Lc_goal = .5
        tc_goal = 5
        test_params = {'num_axial_points':20, "num_time_steps":5, "Lc":Lc_goal, 'tc':tc_goal}
        goal_results = cos_wiggle_fuction(**test_params)


        field_data = convert_dictionary_to_field_data(goal_results, ['x', 'y'])

        model = PythonModel(cos_wiggle_fuction, field_coordinates=['x', 'y'])
        model.add_constants(num_axial_points = test_params['num_axial_points'], num_time_steps = test_params['num_time_steps'])

        poly_order = 4
        max_depth = 4
        my_objective = PolynomialHWDObjective(None, "T", max_depth=max_depth, polynomial_order=poly_order)

        Lc = Parameter("Lc", .1, 1)
        tc = Parameter('tc', 1, 10)
        calibration = GradientCalibrationStudy(Lc, tc)

        calibration.add_evaluation_set(model, my_objective, field_data)
        calibration.set_core_limit(8)
        
        results = calibration.launch()
        L_cal = results.outcome['best:Lc']
        tc_cal = results.outcome['best:tc']

        self.assertAlmostEqual(L_cal, Lc_goal, delta = self._error_tol * Lc_goal)
        self.assertAlmostEqual(tc_cal, tc_goal, delta = self._error_tol * tc_cal)
    
    def test_characteristic_length_and_time_gradient_calibration_x_and_y(self):
        Lc_goal = .5
        tc_goal = 5
        test_params = {'num_axial_points':22, "num_time_steps":5, "Lc":Lc_goal, 'tc':tc_goal}
        goal_results = cos_double_wiggle_fuction(**test_params)


        field_data = convert_dictionary_to_field_data(goal_results, ['x', 'y'])

        model = PythonModel(cos_double_wiggle_fuction, field_coordinates=['x', 'y'])
        model.add_constants(num_axial_points = test_params['num_axial_points'], num_time_steps = test_params['num_time_steps'])

        poly_order = 4
        max_depth = 4
        my_objective = PolynomialHWDObjective(None, "T", max_depth=max_depth, polynomial_order=poly_order)

        Lc = Parameter("Lc", .1, 1)
        tc = Parameter('tc', 1, 10)
        calibration = GradientCalibrationStudy(Lc, tc)

        calibration.add_evaluation_set(model, my_objective, field_data)
        calibration.set_core_limit(8)
        

        results = calibration.launch()
        L_cal = results.outcome['best:Lc']
        tc_cal = results.outcome['best:tc']

        self.assertAlmostEqual(L_cal, Lc_goal, delta = self._error_tol * Lc_goal)
        self.assertAlmostEqual(tc_cal, tc_goal, delta = self._error_tol * tc_cal)


class TwoParameterColocationPythonWaveletCalibrationTest(MatcalUnitTest):

    def setUp(self):
        super().setUp(__file__)
        self._error_tol = 1.0e-3

    def test_characteristic_length_and_time_pattern_search_calibration(self):
        Lc_goal = .5
        tc_goal = 5
        test_params = {'num_axial_points':25, "num_time_steps":5, "Lc":Lc_goal, 'tc':tc_goal}
        goal_params = {'num_axial_points':55, "num_time_steps":5, "Lc":Lc_goal, 'tc':tc_goal}
        goal_results = cos_wiggle_fuction(**goal_params)
        test_results = cos_wiggle_fuction(**test_params)


        field_data = convert_dictionary_to_field_data(goal_results, ['x', 'y'])
        test_field_data = convert_dictionary_to_field_data(test_results, ['x', 'y'])

        model = PythonModel(cos_wiggle_fuction, field_coordinates=['x', 'y'])
        model.add_constants(num_axial_points = test_params['num_axial_points'], num_time_steps = test_params['num_time_steps'])

        poly_order = 4
        max_depth = 4
        my_objective = PolynomialHWDObjective(None, "T", max_depth=max_depth, polynomial_order=poly_order)
        my_objective._set_colocation(test_field_data.skeleton, 2, 2)

        Lc = Parameter("Lc", .1, 1)
        tc = Parameter('tc', 1, 10)
        calibration = PatternSearchCalibrationStudy(Lc, tc)
        calibration.set_solution_target(1e-6)


        calibration.add_evaluation_set(model, my_objective, field_data)
        calibration.set_core_limit(8)
        

        results = calibration.launch()
        L_cal = results.outcome['best:Lc']
        tc_cal = results.outcome['best:tc']
        self._error_tol=1e-2
        self.assertAlmostEqual(L_cal, Lc_goal, delta = self._error_tol * Lc_goal)
        self.assertAlmostEqual(tc_cal, tc_goal, delta = self._error_tol * tc_cal)
    
    def test_characteristic_length_and_time_gradient_calibration_x_and_y(self):
        Lc_goal = .5
        tc_goal = 5
        test_params = {'num_axial_points':25, "num_time_steps":5, "Lc":Lc_goal, 'tc':tc_goal}
        goal_params = {'num_axial_points':55, "num_time_steps":5, "Lc":Lc_goal, 'tc':tc_goal}
        goal_results = cos_double_wiggle_fuction(**goal_params)
        test_results = cos_double_wiggle_fuction(**test_params)


        field_data = convert_dictionary_to_field_data(goal_results, ['x', 'y'])
        test_field_data = convert_dictionary_to_field_data(test_results, ['x', 'y'])

        model = PythonModel(cos_double_wiggle_fuction, field_coordinates=['x', 'y'])
        model.add_constants(num_axial_points = test_params['num_axial_points'], num_time_steps = test_params['num_time_steps'])

        poly_order = 4
        max_depth = 4
        my_objective = PolynomialHWDObjective(None, "T", max_depth=max_depth, polynomial_order=poly_order)
        my_objective._set_colocation(test_field_data.skeleton, 2, 2)

        Lc = Parameter("Lc", .1, 1)
        tc = Parameter('tc', 1, 10)
        calibration = GradientCalibrationStudy(Lc, tc)

        calibration.add_evaluation_set(model, my_objective, field_data)
        calibration.set_core_limit(8)
        

        results = calibration.launch()
        L_cal = results.outcome['best:Lc']
        tc_cal = results.outcome['best:tc']

        self.assertAlmostEqual(L_cal, Lc_goal, delta = self._error_tol * Lc_goal)
        self.assertAlmostEqual(tc_cal, tc_goal, delta = self._error_tol * tc_cal)

