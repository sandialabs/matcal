from copy import deepcopy
from matcal.full_field.data import convert_dictionary_to_field_data
from matcal.full_field.data_importer import mesh_file_to_skeleton
from matcal.full_field.objective import InterpolatedFullFieldObjective, MechanicalVFMObjective, PolynomialHWDObjective
from matcal.full_field.qoi_extractor import ExternalVirtualPowerExtractor, InternalVirtualPowerExtractor
import numpy as np
                                  
from matcal.core.data import DataCollection, ReturnPassedDataConditioner, \
                             convert_dictionary_to_data
from matcal.core.objective import ObjectiveCollection, ObjectiveSet
from matcal.core.objective_results import ObjectiveResults
from matcal.core.state import State

from matcal.core.tests.MatcalUnitTest import MatcalUnitTest


def user_weight_func(indep_var, dep_avr, resids):
    import numpy as np
    weights = indep_var*0+1
    return resids*weights

class TestInterpolatedFullFieldObjective(MatcalUnitTest):
 
    class SpyIFFO(InterpolatedFullFieldObjective):
 
        def get_mesh_skeleton(self):
            return self._make_skeleton_from_mesh_file()
        
        def clean_up(self):
            self._experiment_qoi_extractor.clean_up()

        def spy_interp_poly_order(self):
            return self._polynomial_order
        
        def spy_interp_search_radius(self):
            return self._search_radius_multiplier

    def setUp(self):
        super().setUp(__file__) 
        self._input_files = "/".join([self.get_current_files_path(__file__), "input_files"])

    def test_set_interp_parameters(self):
        iffo, dic_loc, mesh_loc, mesh_sets = self._create_square_objective()
        self.assertAlmostEqual(iffo.spy_interp_poly_order(), 2)
        self.assertAlmostEqual(iffo.spy_interp_search_radius(), 2)
        iffo.set_interpolation_parameters(4, 3)
        self.assertAlmostEqual(iffo.spy_interp_poly_order(), 4)
        self.assertAlmostEqual(iffo.spy_interp_search_radius(), 3)

    def test_return_zeros_from_constants(self):
        iffo, dic_loc, mesh_loc, mesh_sets = self._create_square_objective()

        def fun(x, y, ts, s=1):
            out = np.zeros([len(ts),len(x)])
            for t_i, t in enumerate(ts):
                out[t_i, :] = s + np.power(x,2) + 2*y + t
            return out
        
        times = [0,1,2]
        exp_data_collection = self._make_spatial_data(dic_loc, fun, times, 0)
        sim_times = [0, .25, .75, 1.5, 3.0]
        sim_data_collection = self._make_spatial_data(mesh_loc, fun, sim_times,0, **mesh_sets)
        obj_set = ObjectiveSet(ObjectiveCollection('t', iffo), exp_data_collection, exp_data_collection.states, ReturnPassedDataConditioner)

        results_obj, results_qoi   = obj_set.calculate_objective_set_results(sim_data_collection)
        residual = results_obj[iffo.name].calibration_residuals
        obj = results_obj[iffo.name].get_objective()
        n_nodes = 9
        n_time = len(times)
        n_dof = 2
        self.assertEqual(len(residual), n_nodes * n_time * n_dof)
        self.assertAlmostEqual(np.sum(np.abs(residual)), 0)
        self.assertAlmostEqual(obj, 0)

    def _make_spatial_data(self, locations, fun, times, offset, **surfaces):
        raw_data = {'time': times, "displacement_x":fun(locations['x'], locations['y'], times, offset), "displacement_y":fun(locations['x'], locations['y'], times, offset)}
        raw_data.update(locations)
        raw_data['surfaces'] = surfaces
        data = convert_dictionary_to_field_data(raw_data, ['x', 'y'], node_set_name='surfaces')
        data.set_name('first')
        new_data_collection = DataCollection('constant', data)
        return new_data_collection

    def _create_square_objective(self):
        mesh_file = f"{self._input_files}/interpolation_test.json"
        mesh_skeleton = mesh_file_to_skeleton(mesh_file)
        mesh_loc = {'x':mesh_skeleton.spatial_coords[:,0], 'y':mesh_skeleton.spatial_coords[:,1]}
        fem_surface = "DIC"
        n_points = 40
        dic_loc = {'x': np.random.uniform(0,1, n_points), 'y': np.random.uniform(0,1, n_points)}
        time_variable = 'time'
        dep_vars = ['displacement_x', 'displacement_y']
        iffo = self.SpyIFFO(mesh_file, *dep_vars, fem_surface=fem_surface)
        return iffo, dic_loc, mesh_loc, mesh_skeleton.surfaces

    def test_return_unit(self):
        iffo, dic_loc, mesh_loc, mesh_sets = self._create_square_objective()

        def fun(x, y, ts, s=1):
            out = np.zeros([len(ts),len(x)])
            for t_i, t in enumerate(ts):
                out[t_i, :] = s + np.power(x,2) + 2*y + t
            return out
        
        times = [0,1,2]
        exp_data_collection = self._make_spatial_data(dic_loc, fun, times, 0)
        sim_times = [0, .25, .75, 1.5, 3.0]
        sim_data_collection = self._make_spatial_data(mesh_loc, fun, sim_times, 1, **mesh_sets)
        obj_set = ObjectiveSet(ObjectiveCollection('t', iffo), exp_data_collection, exp_data_collection.states, ReturnPassedDataConditioner)

        results_obj, results_qoi   = obj_set.calculate_objective_set_results(sim_data_collection)
        residual = results_obj[iffo.name].calibration_residuals
    
        obj = results_obj[iffo.name].get_objective()
        n_nodes = 9
        n_time = len(times)
        n_dof = 2
        self.assertEqual(len(residual), n_nodes * n_time * n_dof)
        rx = 1/np.sqrt(n_nodes*n_time)
        ry = 1/np.sqrt(n_nodes*n_time)
        r_one_step = [rx,rx,rx,rx,rx,rx,rx,rx,rx,ry,ry,ry,ry,ry,ry,ry,ry,ry]
        goal  = np.concatenate([r_one_step, r_one_step, r_one_step])
        self.assert_close_arrays(residual, goal)
        self.assertAlmostEqual(obj, np.linalg.norm(goal)**2)

    def test_return_linear_two_data_sets(self):
        iffo, dic_loc, mesh_loc, mesh_sets = self._create_square_objective()

        def fun(x, y, ts, s=1):
            out = np.zeros([len(ts),len(x)])
            for t_i, t in enumerate(ts):
                out[t_i, :] = s + np.power(x,2) + 2*y + t
            return out
        
        times = [0,1,2]
        exp_data_collection2 = self._make_spatial_data(dic_loc, fun, times, 0)
        exp_data_collection1 = self._make_spatial_data(dic_loc, fun, times, 0)
        lookup_state = list(exp_data_collection2.states.values())[0]
        state_to_set = list(exp_data_collection1.states.values())[0]
        data_to_add = deepcopy(exp_data_collection2[lookup_state][0])
        data_to_add.set_name('aux_data')
        data_to_add.set_state(state_to_set)
        exp_data_collection1.add(data_to_add)

        sim_times = [0, .25, .75, 1.5, 3.0]
        sim_data_collection = self._make_spatial_data(mesh_loc, fun, sim_times, 0, **mesh_sets)
        obj_set = ObjectiveSet(ObjectiveCollection('t', iffo), exp_data_collection1, exp_data_collection1.states, ReturnPassedDataConditioner)

        results_obj, results_qoi = obj_set.calculate_objective_set_results(sim_data_collection)
        residual = results_obj[iffo.name].calibration_residuals

        self.assertAlmostEqual(np.linalg.norm(residual), 0)

    def _make_dcs_and_obj(self, field_func, mesh_file, fem_surface, exp_time, sim_time, exp_x, exp_y, init_data_specific=True):
        exp_data_0 = {'x':np.array(exp_x), 'y':np.array(exp_y), 'time':np.array(exp_time)}
        exp_data_1 = {}
        for key, value in exp_data_0.items():
            exp_data_1[key] = np.multiply(value, np.random.uniform(.9,1.1, value.shape))
        exp_data_0['T'] = field_func(exp_data_0['x'], exp_data_0['y'], exp_data_0['time'])
        exp_data_1['T'] = field_func(exp_data_1['x'], exp_data_1['y'], exp_data_1['time'])
        exp_dc = DataCollection('Exp', convert_dictionary_to_field_data(exp_data_0, ['x', 'y']), convert_dictionary_to_field_data(exp_data_1, ['x', 'y']))


        iffo = InterpolatedFullFieldObjective(mesh_file, 'T', fem_surface=fem_surface)
        if init_data_specific:
            iffo.data_specific_initialization(exp_dc)
        sim_extractors = iffo.simulation_qoi_extractor._extractors
        first_key = list(sim_extractors.keys())[0]
        mesh_geo = mesh_file_to_skeleton(mesh_file)
        sim_data = {'x':mesh_geo.spatial_coords[:,0], 'y':mesh_geo.spatial_coords[:,1], 'surf':mesh_geo.surfaces, 'time':np.array(sim_time)}
        sim_data['T'] = field_func(sim_data['x'], sim_data['y'], sim_data['time'])
        sim_data = convert_dictionary_to_field_data(sim_data, ['x', 'y'], node_set_name='surf')
        sim_dc = DataCollection('Sim', sim_data)
        return exp_dc,iffo,sim_dc

    def _get_residual(self, exp_dc, iffo, sim_dc):
        sim_qoi = DataCollection('QOI:SIM')
        for state in exp_dc.states:
            for ref_data  in exp_dc[state]:
                for work_data in sim_dc[state]:
                    qoi = iffo.simulation_qoi_extractor.calculate(work_data, ref_data, 'T')
                    sim_qoi.add(qoi)
       
        exp_qoi = DataCollection('QOI:EXP') 
        for state in exp_dc.states:
            for ref_data  in sim_dc[state]:
                for work_data in exp_dc[state]:
                    qoi = iffo.experiment_qoi_extractor.calculate(work_data, ref_data, 'T')
                    exp_qoi.add(qoi)
        results = iffo.calculate(exp_qoi, exp_qoi, sim_qoi, sim_qoi)
        residual  = results.calibration_residuals
        return residual


class TestPolynomialHWDObjective(MatcalUnitTest):

    def setUp(self):
        super().setUp(__file__)

    def test_get_zero_for_same_data(self):
        max_depth = 1
        max_poly = 2
        time_var = 'time'
        dep_var = 'temp'
        raw_data1 = {'x':np.array([-1.1,-.1, -.9, 1.1, 1., .9]), 'y':np.array([1, 1.1, 1.2, 1, 1.1, 1.2]), time_var:np.array([1,2,3])}
        raw_data1[dep_var] = np.ones([len(raw_data1[time_var]), len(raw_data1['x'])])
        data1 = convert_dictionary_to_field_data(raw_data1,['x', 'y'])
        dc_all =DataCollection('all', data1)
        dc_all_copy = deepcopy(dc_all)

        hwdo = PolynomialHWDObjective(dep_var, max_depth, max_poly)
        hwdo.data_specific_initialization(dc_all)
        results = hwdo.calculate(dc_all, dc_all, dc_all_copy, dc_all_copy)
        resudial  = results.flattened_calibration_residuals
        self.assert_close_arrays(np.zeros_like(resudial), resudial)

    def test_get_zero_for_same_data(self):
        max_depth = 1
        max_poly = 2
        time_var = 'time'
        dep_var = 'temp'
        raw_data1 = {'x':np.array([-1.1,-.1, -.9, 1.1, 1., .9]), 'y':np.array([1, 1.1, 1.2, 1, 1.1, 1.2]), time_var:np.array([1,2,3])}
        raw_data1[dep_var] = np.ones([len(raw_data1[time_var]), len(raw_data1['x'])])
        data1 = convert_dictionary_to_field_data(raw_data1,['x', 'y'])
        dc_all =DataCollection('all', data1)
        dc_all_copy = deepcopy(dc_all)

        hwdo = PolynomialHWDObjective(None, dep_var, max_depth=max_depth, polynomial_order=max_poly)
        hwdo.data_specific_initialization(dc_all)
        results_obj, results_qoi = hwdo.calculate(dc_all, dc_all, dc_all_copy, dc_all_copy)
        residual  = results_obj.calibration_residuals
        self.assert_close_arrays(np.zeros_like(residual), residual)

    def _linear(self, xs, ys, times):
        value = np.zeros([len(times), len(xs)])
        for row_i, t in enumerate(times):
            for col_i, (x, y) in enumerate(zip(xs, ys)):
                value[row_i, col_i] = x + y + t  
        return value

    def test_get_zero_for_same_data_different_time_and_location(self):
        max_depth = 1
        max_poly = 2
        dep_var = 'temp'

        raw_data1 = {'x':np.array([-1.1,-.1, -.9, 1.1, 1., .9]), 'y':np.array([1, 1.1, 1.2, 1, 1.1, 1.2]), 'time':np.array([1,2,3])}
        raw_data1['temp'] = self._linear(raw_data1['x'], raw_data1['y'], raw_data1['time'])
        data1 = convert_dictionary_to_field_data(raw_data1,['x', 'y'])

        raw_data2 = {'x':np.array([-1.2,-1., -.85, 1.05, 1.02, .93]), 'y':np.array([1.02, 1.08, 1.17, 1.01, 1.11, 1.21]), 'time':np.array([.5,1.5,2.5,3.5])}
        raw_data2['temp'] = self._linear(raw_data2['x'], raw_data2['y'], raw_data2['time'])
        data2 = convert_dictionary_to_field_data(raw_data2,['x', 'y'])

        dc_exp = DataCollection('exp', data1)
        dc_sim = DataCollection('sim', data2)

        hwdo = PolynomialHWDObjective(None, dep_var, max_depth=max_depth, polynomial_order=max_poly)
        ob_set = ObjectiveSet(ObjectiveCollection("test", hwdo), dc_exp, dc_exp.states)

        results_obj, results_qoi  = ob_set.calculate_objective_set_results(dc_sim)
        results = results_obj[hwdo.name]
        resudial  = results.calibration_residuals
        self.assert_close_arrays(np.zeros_like(resudial), resudial)

    def test_get_zero_for_same_data_different_time_and_location_two_states(self):
        max_depth = 1
        max_poly = 2
        time_var = 'time'
        dep_var = 'temp'

        state1 = State('1')
        state2 = State('2')

        raw_data11 = {'x':np.array([-1.1,-.1, -.9, 1.1, 1., .9]), 'y':np.array([1, 1.1, 1.2, 1, 1.1, 1.2]), 'time':np.array([1,2,3])}
        raw_data11['temp'] = self._linear(raw_data11['x'], raw_data11['y'], raw_data11['time'])
        data11 = convert_dictionary_to_field_data(raw_data11,['x', 'y'])
        data11.set_state(state1)

        raw_data12 = {'x':np.array([-1.1,-.1, -.9, 1.1, 1., .9]), 'y':np.array([1, 1.1, 1.2, 1, 1.1, 1.2]), 'time':np.array([1,2,3])}
        raw_data12['temp'] = self._linear(raw_data12['x'], raw_data12['y'], raw_data12['time']) * 2
        data12 = convert_dictionary_to_field_data(raw_data12,['x', 'y'])
        data12.set_state(state2)


        raw_data21 = {'x':np.array([-1.2,-1., -.85, 1.05, 1.02, .93]), 'y':np.array([1.02, 1.08, 1.17, 1.01, 1.11, 1.21]), 'time':np.array([.5,1.5,2.5,3.5])}
        raw_data21['temp'] = self._linear(raw_data21['x'], raw_data21['y'], raw_data21['time'])
        data21 = convert_dictionary_to_field_data(raw_data21,['x', 'y'])
        data21.set_state(state1)

        raw_data22 = {'x':np.array([-1.2,-1., -.85, 1.05, 1.02, .93]), 'y':np.array([1.02, 1.08, 1.17, 1.01, 1.11, 1.21]), 'time':np.array([.5,1.5,2.5,3.5])}
        raw_data22['temp'] = self._linear(raw_data22['x'], raw_data22['y'], raw_data22['time']) *2
        data22 = convert_dictionary_to_field_data(raw_data22,['x', 'y'])
        data22.set_state(state2)

        dc_exp = DataCollection('exp', data11, data12)
        dc_sim = DataCollection('sim', data21, data22)

        hwdo = PolynomialHWDObjective(None, dep_var, max_depth=max_depth, polynomial_order=max_poly)
        ob_set = ObjectiveSet(ObjectiveCollection("test", hwdo), dc_exp, dc_exp.states)

        results_obj, results_qoi = ob_set.calculate_objective_set_results(dc_sim)
        results = results_obj[hwdo.name]
        resudial  = results.calibration_residuals
        self.assert_close_arrays(np.zeros_like(resudial), resudial)

    def get_random_points(self, n_points, n_dim):
        p_array = np.random.uniform(0, 1, [n_points, n_dim])
        dim_names = ['x', 'y', 'z']
        points = {}
        for i in range(n_dim):
            points[dim_names[i]] = p_array[:,i]
        return points

    def test_extract_data_from_surface_linear(self):
        max_depth = 0
        max_poly = 2
        time_var = 'time'
        dep_var = "temp"
        s_name = "TeSt_SuRfAcE"

        ref_dict = self.get_random_points(20,2)
        test_dict = self.get_random_points(30,3)
        test_dict['node_surf'] = {s_name:list(range(0,30,2))}

        ref_dict[time_var]= np.linspace(0, 10, 5)
        test_dict[time_var] = np.linspace(0, 10, 8)

        def linear_fun(x, y, t):
            out = np.zeros([len(t), len(x)])
            pos = x + 2 * y
            for t_idx, t_val  in enumerate(t):
                out[t_idx, :] = pos + t_val
            return out
        
        for data_dict in [ref_dict, test_dict]:
            data_dict[dep_var] = linear_fun(data_dict['x'], data_dict['y'], data_dict[time_var])
        ref = convert_dictionary_to_field_data(ref_dict, ['x','y'])
        test = convert_dictionary_to_field_data(test_dict, ['x', 'y', 'z'], node_set_name='node_surf')
        
        dc_exp = DataCollection('exp', ref)
        dc_sim = DataCollection('sim', test)

        hwdo = PolynomialHWDObjective( None, dep_var, max_depth=max_depth, polynomial_order=max_poly)
        hwdo.extract_data_from_mesh_surface(s_name)
        obj_set = ObjectiveSet(ObjectiveCollection('test', hwdo), dc_exp, dc_exp.states)
        results_obj, results_qoi   = obj_set.calculate_objective_set_results(dc_sim)
        results = results_obj[hwdo.name]
        residual = results.calibration_residuals
        self.assert_close_arrays(np.zeros_like(residual), residual)

    def test_extract_data_from_surface_linear_colocate(self):
        max_depth = 0
        max_poly = 2
        time_var = 'time'
        dep_var = "temp"
        s_name = "TeSt_SuRfAcE"

        exp_dict = self.get_random_points(20,2)
        sim_dict = self.get_random_points(30,3)
        sim_dict['node_surf'] = {s_name:list(range(0,30,2))}

        exp_dict[time_var]= np.linspace(0, 10, 5)
        sim_dict[time_var] = np.linspace(0, 10, 8)

        def linear_fun(x, y, t):
            out = np.zeros([len(t), len(x)])
            pos = x + 2 * y
            for t_idx, t_val  in enumerate(t):
                out[t_idx, :] = pos + t_val
            return out
        
        for data_dict in [exp_dict, sim_dict]:
            data_dict[dep_var] = linear_fun(data_dict['x'], data_dict['y'], data_dict[time_var])
        exp = convert_dictionary_to_field_data(exp_dict, ['x','y'])
        sim = convert_dictionary_to_field_data(sim_dict, ['x', 'y', 'z'], node_set_name='node_surf')
        
        dc_exp = DataCollection('exp', exp)
        dc_sim = DataCollection('sim', sim)

        hwdo = PolynomialHWDObjective(sim.skeleton, dep_var,max_depth=max_depth, polynomial_order=max_poly)
        hwdo.extract_data_from_mesh_surface(s_name)
        obj_set = ObjectiveSet(ObjectiveCollection('test', hwdo), dc_exp, dc_exp.states)
        results_obj, results_qoi   = obj_set.calculate_objective_set_results(dc_sim)
        results = results_obj[hwdo.name]
        residual = results.calibration_residuals
        self.assert_close_arrays(np.zeros_like(residual), residual)

    def test_extract_data_from_surface_trig_colocate(self):
        time_var = 'time'
        dep_var = "temp"
        s_name = "TeSt_SuRfAcE"

        n_exp = 900
        n_sim = 450
        exp_dict = self.get_random_points(n_exp,2)
        sim_dict = self.get_random_points(n_sim,3)
        sim_dict['node_surf'] = {s_name:list(range(0,n_sim,2))}

        exp_dict[time_var]= np.linspace(0, 10, 5)
        sim_dict[time_var] = np.linspace(0, 10, 8)

        def trial_fun(x, y, t):
            out = np.zeros([len(t), len(x)])
            pos = np.sin(np.pi * (x + y ))
            for t_idx, t_val  in enumerate(t):
                out[t_idx, :] = pos + t_val
            return out
        
        for data_dict in [exp_dict, sim_dict]:
            data_dict[dep_var] = trial_fun(data_dict['x'], data_dict['y'], data_dict[time_var])
        exp = convert_dictionary_to_field_data(exp_dict, ['x','y'])
        sim = convert_dictionary_to_field_data(sim_dict, ['x', 'y', 'z'], node_set_name='node_surf')
        
        dc_exp = DataCollection('exp', exp)
        dc_sim = DataCollection('sim', sim)

        hwdo = PolynomialHWDObjective(sim, dep_var,max_depth=4, polynomial_order=4)
        hwdo.extract_data_from_mesh_surface(s_name)

        obj_set = ObjectiveSet(ObjectiveCollection('test', hwdo), dc_exp, dc_exp.states)
        results_obj, results_qoi   = obj_set.calculate_objective_set_results(dc_sim)
        results = results_obj[hwdo.name]
        residual = results.calibration_residuals
        self.assert_close_arrays(np.zeros_like(residual), residual, atol=1e-4)

class TestMechanicalVFMObjective(MatcalUnitTest):
    class MechanicalVFMObjectiveSpy(MechanicalVFMObjective):
        
        def confirm_gradient(self, goal_arrays):
            okay = True
            for i_center in range(np.shape(goal_arrays)[0]):
                goal = goal_arrays[i_center, :, :]
                test = self._virtual_gradient[i_center, :, :]
                diff = test - goal
                max_diff = np.max(np.abs(diff.flatten()))
                okay = okay and max_diff < 1e-8
            return okay

        def _close_floats(self, a, b):
            diff = np.abs(a - b)
            delta = 1e-8
            return diff < delta

        def get_experiment_qoi_extractor(self):
            return self._experiment_qoi_extractor

        def get_simulation_qoi_extractor(self):
            return self._simulation_qoi_extractor

    def setUp(self):
        super().setUp(__file__)
        self.thickness = 1
        def const_vel(locations):
            vel = np.zeros([len(locations[:, 0]), 2])
            vel[:, 0] = 1
            vel[:, 1] = 1
            return vel

    def test_init(self):
        obj = MechanicalVFMObjective("time")
        with self.assertRaises(obj.TypeError):
            MechanicalVFMObjective(1)
        with self.assertRaises(obj.TypeError):
            MechanicalVFMObjective("yay",1)
    
    def test_get_results_object_from_calculate(self):
        obj = MechanicalVFMObjective()
                                               
        dc_zero = DataCollection("zeros", self.make_mock_exo_zero_data())
        dc_fdata = DataCollection("f data", self.make_mock_exo_zero_data_coarse_time())

        obj_set = ObjectiveSet(ObjectiveCollection("test", obj), dc_fdata, dc_fdata.states)

        results_obj, results_qoi   = obj_set.calculate_objective_set_results(dc_zero)
        self.assertIsInstance(results_obj[obj.name],
            ObjectiveResults)

    def test_get_zeros_from_calculate_with_zeros(self):
        exp_data = self.make_mock_exo_zero_data_coarse_time()
        sim_data = self.make_mock_exo_zero_data()

        obj = MechanicalVFMObjective()

        dc_zero = DataCollection("zeros", sim_data)
        dc_fdata = DataCollection("f data", exp_data) 

        obj_set = ObjectiveSet(ObjectiveCollection("test", obj), dc_fdata, dc_fdata.states)
        results_obj   = obj_set.calculate_objective_set_results(dc_zero)[0][obj.name]
        goal = np.zeros(3)

        
        self.assert_close_arrays(goal, results_obj.get_flattened_weighted_conditioned_residuals(), 1e-7)

    def test_get_exp_array_from_calculate_with_zero_stress(self):
        exp_data = self._make_linear_field_values_at_time_coarse_time()
        sim_data = self.make_mock_exo_zero_data()

        obj = MechanicalVFMObjective()
        obj.set_as_large_data_sets_objective(False)

        dc_zero = DataCollection("zeros", sim_data)
        dc_fdata = DataCollection("f data", exp_data) 

        obj_set = ObjectiveSet(ObjectiveCollection("test", obj), dc_fdata, dc_fdata.states)
        results_obj = obj_set.calculate_objective_set_results(dc_zero)[0][obj.name]
        goal = np.array([0, -100, -200])
        for resid_i in range(len(goal)):
            self.assertAlmostEqual(goal[resid_i], results_obj.residuals["matcal_default_state"][0]["virtual_power"][resid_i])

        goal = np.array([0, -0.5, -1])
        self.assert_close_arrays(goal, results_obj.get_flattened_weighted_conditioned_residuals(), 1e-7)

    def test_zeros_from_calculate_with_non_zero_data(self):
        exp_data = self.make_mock_exo_zero_data_coarse_time()
        sim_data = self.make_mock_exo_zero_data()

        obj = MechanicalVFMObjective()

        dc_linear = DataCollection("zeros", sim_data)
        dc_fdata = DataCollection("f data", exp_data) 
        goal = np.array([0, -0, -0])

        obj_set = ObjectiveSet(ObjectiveCollection("test", obj), dc_fdata, dc_fdata.states)
        results_obj  = obj_set.calculate_objective_set_results(dc_linear)[0][obj.name]
        self.assert_close_arrays(goal, results_obj.get_flattened_weighted_conditioned_residuals(), 1e-7)

    def test_confirm_correct_default_qoi_extractor(self):
        obj_spy = self.MechanicalVFMObjectiveSpy()

        self.assertIsInstance(obj_spy.get_experiment_qoi_extractor(), ExternalVirtualPowerExtractor)
        self.assertIsInstance(obj_spy.get_simulation_qoi_extractor(), InternalVirtualPowerExtractor)

    @staticmethod
    def make_mock_exo_zero_data():
        n_cells = 6
        
        x_locs = [0, 0, 1, 1, 2, 2]
        y_locs = [0, 1, 0, 1, 0, 1]
        times = np.array([0, .75, 1.5, 2.0])
        n_times = len(times)
        ones = np.ones((n_times, n_cells))
        data_dict = {"time":times, "x":x_locs, "y":y_locs, 
                    "element_thickness":ones, 
                    "element_area":ones,
                    "first_pk_stress_xx":np.outer((np.zeros(n_cells)), times).T, 
                    "first_pk_stress_yy":np.outer((np.zeros(n_cells)), times).T,
                    "first_pk_stress_xy":np.outer((np.zeros(n_cells)), times).T,
                    "first_pk_stress_yx":np.outer((np.zeros(n_cells)), times).T, 
                    "centroid_x":[x_locs]*n_times,
                    "centroid_y":[y_locs]*n_times}
        data = convert_dictionary_to_field_data(data_dict, ["x", "y"])

        return data

    @staticmethod
    def make_mock_exo_zero_data_coarse_time():
        n_cells = 6
        
        x_locs = [0, 0, 1, 1, 2, 2]
        y_locs = [0, 1, 0, 1, 0, 1]
        times = np.array([0, 1, 2.0])
        n_times = len(times)
        ones = np.ones((n_times, n_cells))
        data_dict = {"time":times, "x":x_locs, "y":y_locs, 
                    "element_thickness":ones, 
                    "element_area":ones,
                    "first_pk_stress_xx":np.outer((np.zeros(n_cells)), times).T, 
                    "first_pk_stress_yy":np.outer((np.zeros(n_cells)), times).T,
                    "first_pk_stress_xy":np.outer((np.zeros(n_cells)), times).T,
                    "first_pk_stress_yx":np.outer((np.zeros(n_cells)), times).T, 
                    "centroid_x":[x_locs]*n_times,
                    "centroid_y":[y_locs]*n_times, 
                    "load":times*0, 
                    "displacement":times*.02}
        data = convert_dictionary_to_field_data(data_dict, ["x", "y"])

        return data

    @staticmethod
    def _make_linear_field_values_at_time():
        area = 2
        n_cells = 4
        x_locs = [0, 0, 1, 1, 2, 2]
        y_locs = [0, 1, 0, 1, 0, 1]
        times = np.array([0, .75, 1.5, 2.0])
        _a = 100/4* times / area
        n_times = len(times)
        ones = np.ones((n_times, n_cells))
        data_dict = {"time":times, "x":x_locs, "y":y_locs, 
                    "element_thickness":ones, 
                    "element_area":ones*area,
                    "first_pk_stress_xx":np.outer(_a, np.ones(n_cells)), 
                    "first_pk_stress_yy":np.outer(_a, np.ones(n_cells)),
                    "first_pk_stress_xy":np.outer(_a, np.ones(n_cells)),
                    "first_pk_stress_yx":np.outer(_a, np.ones(n_cells)), 
                    "centroid_x":[x_locs]*n_times,
                    "centroid_y":[y_locs]*n_times}
        data = convert_dictionary_to_field_data(data_dict, ["x", "y"])
        return data

    @staticmethod
    def _make_linear_field_values_at_time_coarse_time():
        area = 2
        n_cells = 4
        x_locs = [0, 0, 1, 1, 2, 2]
        y_locs = [0, 1, 0, 1, 0, 1]
        times = np.array([0, 1, 2])
        _a = 100/4* times / area
        n_times = len(times)
        ones = np.ones((n_times, n_cells))
        data_dict = {"time":times, "x":x_locs, "y":y_locs, 
                    "element_thickness":ones, 
                    "element_area":ones*area,
                    "first_pk_stress_xx":np.outer(_a, np.ones(n_cells)), 
                    "first_pk_stress_yy":np.outer(_a, np.ones(n_cells)),
                    "first_pk_stress_xy":np.outer(_a, np.ones(n_cells)),
                    "first_pk_stress_yx":np.outer(_a, np.ones(n_cells)), 
                    "centroid_x":[x_locs]*n_times,
                    "centroid_y":[y_locs]*n_times,
                    "load":times*100, 
                    "displacement":times*0.02}
        data = convert_dictionary_to_field_data(data_dict, ["x", "y"])
        return data
