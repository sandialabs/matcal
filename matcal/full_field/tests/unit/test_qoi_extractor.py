from matcal.core.data import convert_dictionary_to_data
from matcal.core.tests.MatcalUnitTest import MatcalUnitTest
from matcal.full_field.TwoDimensionalFieldGrid import (MeshSkeleton, NewToOldRemapper, 
    _get_node_coordinate_mapping, _get_surface_nodes)
from matcal.full_field.data import convert_dictionary_to_field_data
import numpy as np

from matcal.full_field.qoi_extractor import (ExternalVirtualPowerExtractor, 
    FieldInterpolatorExtractor, FieldTimeInterpolatorExtractor, 
    FlattenFieldDataExtractor, HWDColocatingExperimentSurfaceExtractor, 
    HWDExperimentSurfaceExtractor, HWDPolynomialSimulationSurfaceExtractor, 
    InternalVirtualPowerExtractor, MeshlessSpaceInterpolatorExtractor)
from matcal.full_field.qoi_extractor import (_default_velocity_gradient_function, 
    _default_velocity_function)


class TestFieldDataFlattener(MatcalUnitTest):

    def setUp(self):
        super().setUp(__file__)

    def test_flatten_two_dim_array(self):
        fields = ['a','b']
        np.random.seed(123456)
        data_dict = {'a':np.ones([5,4]), 'b':np.random.random([5,4])}
        fake_qoi = convert_dictionary_to_data(data_dict)
        extractor = FlattenFieldDataExtractor()
        qoi = extractor.calculate(fake_qoi, None, fields)
        for f in fields: 
            self.assert_close_arrays(qoi[f], data_dict[f].flatten())

class TestFieldInterpolatorDataExtractor(MatcalUnitTest):

    class FieldInterpolatorExtractorSpy(FieldInterpolatorExtractor):
        pass

    def setUp(self):
        super().setUp(__file__)

    def test_calculate_correctly_returns_constant(self):
        grid = self.make_one_element_grid()
        cloud_points = np.array([[0, 1, -1],[0, 1, -1]]).T
        extractor = FieldInterpolatorExtractor(grid, cloud_points, 't')
        field_to_interp = "A"
        dict_to_interp = {field_to_interp: np.array([[3,3,3,3, -10, -10, -10, -10]]), 't':[0]}
        data_to_interp = self.make_field_data(dict_to_interp)
        ref_data = convert_dictionary_to_data({'t':[0]})
        results = extractor.calculate(data_to_interp, ref_data, field_to_interp)
        goal = np.array([[3,3,3]])
        self.assert_close_arrays(results[field_to_interp], goal)

    def test_calculate_correctly_returns_constant_and_linear(self):
        grid = self.make_one_element_grid()
        cloud_points = np.array([[0, 1, -1],[0, 1, -1]]).T
        extractor = FieldInterpolatorExtractor(grid, cloud_points, 't')
        fields_to_interp = ["A", "B"]
        dict_to_interp = {fields_to_interp[0]: np.array([[3,3,3,3, -10, -10, -10, -10]]), 
            fields_to_interp[1]: np.array([[-2, 0, 2, 0, -10, -10, -10, -10]]), 't':[0]}
        data_to_interp = self.make_field_data(dict_to_interp)
        ref_data = convert_dictionary_to_data({'t':[0 ]})
        results = extractor.calculate(data_to_interp, ref_data, fields_to_interp)
        goal = {fields_to_interp[0]:np.array([[3,3,3]]), fields_to_interp[1]:np.array([[0, 2, -2]])}
        self.assert_close_arrays(results["A"], goal["A"])
        self.assert_close_arrays(results["B"], goal["B"])

    def test_calculate_over_multiple_matching_times(self):
        grid = self.make_one_element_grid()
        cloud_points = np.array([[0, 1, -1],[0, 1, -1]]).T
        extractor = FieldInterpolatorExtractor(grid, cloud_points, 't')
        field_to_interp = "A"
        dict_to_interp = {field_to_interp: np.array([[3,3,3,3, -10, -10, -10, -10],
            [-2, 0, 2, 0, -10, -10, -10, -10]]), 't':[0 , 1]}
        data_to_interp = self.make_field_data(dict_to_interp)
        ref_data = convert_dictionary_to_data({'t':[0 , 1]})
        results = extractor.calculate(data_to_interp, ref_data, field_to_interp)
        goal = np.array([[3,3,3], [0, 2, -2]]).flatten()
        self.assert_close_arrays(results[field_to_interp], goal)

    def test_calculate_over_multiple_different_times(self):
        grid = self.make_one_element_grid()
        cloud_points = np.array([[0, 1, -1],[0, 1, -1]]).T
        extractor = FieldInterpolatorExtractor(grid, cloud_points, 't')
        field_to_interp = "A"
        dict_to_interp = {field_to_interp: np.array([[1,10,30,3, -10, -10, -10, -10],
            [6, 10, 50, -2, -10, -10, -10, -10]]), 't':[0 , 1]}
        data_to_interp = self.make_field_data(dict_to_interp)
        ref_data = convert_dictionary_to_data({'t':[.2 , .8]})
        results = extractor.calculate(data_to_interp, ref_data, field_to_interp)
        goal = np.array([[12,34,2], [15, 46, 5]]).flatten()
        self.assert_close_arrays(results[field_to_interp], goal)

    def test_calculate_over_multiple_different_times_and_fields(self):
        grid = self.make_one_element_grid()
        cloud_points = np.array([[0, 1, -1],[0, 1, -1]]).T
        extractor = FieldInterpolatorExtractor(grid, cloud_points, 't')
        dict_to_interp = {"A": np.array([[1,10,30,3, -10, -10, -10, -10], 
            [6, 10, 50, -2, -10, -10, -10, -10]]), 
            "NA": -np.array([[1,10,30,3, -10, -10, -10, -10],
            [6, 10, 50, -2, -10, -10, -10, -10]]), 't':[0 , 1]}
        data_to_interp = self.make_field_data(dict_to_interp)
        ref_data = convert_dictionary_to_data({'t':[.2 , .8]})
        results = extractor.calculate(data_to_interp, ref_data, ["A", "NA"])
        goal = np.array([[12,34,2], [15, 46, 5]]).flatten()
        self.assert_close_arrays(results["A"], goal)
        self.assert_close_arrays(results["NA"], -goal)

    def make_field_data(self, dict_to_interp):
        data_to_interp = convert_dictionary_to_field_data(dict_to_interp)
        results = self.make_one_element_cube_different_surface_numbering()
        data_to_interp._graph, s_map, i_map = results
        for field in data_to_interp.field_names:
            value = data_to_interp[field]
            if value.ndim > 1:
                data_to_interp[field] = value[:,s_map]
        return data_to_interp

    def test_required_points_property_is_number_of_nodes(self):
        grid = self.make_one_element_grid()
        cloud_points = np.array([[0, 1, -1],[0, 1, -1]]).T
        extractor = FieldInterpolatorExtractor(grid, cloud_points, 't')
        self.assertEqual(extractor.number_of_nodes_required, grid.spatial_coords.shape[0])

    def test_extract_subset_of_working_data_from_refernce_grid(self):
        old_grid = self.make_one_element_grid()
        cube = self.make_one_element_cube_same_node_numbering()
        nodes = _get_surface_nodes(cube, old_grid.subset_name)
        self.assert_close_arrays(nodes, [0,1,2,3])

    def test_map_extracted_subset_identity(self):
        old_grid = self.make_one_element_grid()
        cube = self.make_one_element_cube_same_node_numbering()
        mapped_nodes = _get_node_coordinate_mapping(cube, old_grid, old_grid.subset_name)
        self.assert_close_arrays(mapped_nodes, [0,1,2,3])
        self.assert_close_arrays(cube.spatial_coords[mapped_nodes,:2], old_grid.spatial_coords)

    def test_map_extracted_subset_mixed(self):
        old_grid = self.make_one_element_grid()
        cube = self.make_one_element_cube_different_surface_numbering()[0]
        mapped_nodes = _get_node_coordinate_mapping(cube, old_grid, old_grid.subset_name)
        self.assert_close_arrays(cube.spatial_coords[mapped_nodes,:2], old_grid.spatial_coords)

    def make_one_element_grid(self):
        grid = MeshSkeleton()
        grid.spatial_coords = np.array([[-1, 1, 1, -1], [-1, -1, 1, 1]]).T
        grid.connectivity = np.array([[0, 1, 2, 3]])
        grid.subset_name = "front"
        return grid

    def make_one_element_cube_same_node_numbering(self):
        grid = MeshSkeleton()
        grid.spatial_coords = np.array([[-1, 1, 1, -1, -1, 1, 1, -1], 
            [-1, -1, 1, 1, -1, -1, 1, 1], [1, 1, 1, 1, -1, -1, -1, -1]]).T
        grid.connectivity = np.array([[0, 1, 2, 3], [7, 6, 5, 4], [1, 5, 6, 2], 
            [2, 6, 7 , 3], [0, 3, 7 ,4], [0, 4, 5, 1]])
        grid.add_node_sets(front=np.array([0, 1, 2, 3]))
        return grid

    def make_one_element_cube_different_surface_numbering(self):
        grid = self.make_one_element_cube_same_node_numbering()
        shuffle_map = np.array(list(range(8)))
        np.random.seed(123456)
        np.random.shuffle(shuffle_map)
        inverse_map = np.argsort(shuffle_map)
        grid.spatial_coords = grid.spatial_coords[shuffle_map, :]
        for ele in range(grid.connectivity.shape[0]):
            for node_count in range(grid.connectivity.shape[1]):
                grid.connectivity[ele, node_count] = inverse_map[grid.connectivity[ele, node_count]]
        for name in grid.surfaces.keys():
            for node_count in range(len(grid.surfaces[name])):
                grid.surfaces[name][node_count] = inverse_map[grid.surfaces[name][node_count]]
        return grid, shuffle_map, inverse_map


def return_zeros_virtual_velocity_gradient(points, data):
    velocity = np.zeros((np.shape(points)[0],2,2))
    return velocity


def return_threes_virtual_velocity_gradient(points, data):
    velocity = np.ones((np.shape(points)[0],2,2))*3
    return velocity


def return_linear_virtual_velocity_gradient(points, data):
    num_points = np.shape(points)[0]
    velocity = np.ones((num_points,2,2))
    for point_i in range(num_points):
        velocity[point_i, :, :] = -(point_i+1)*np.ones((2,2))*3
    
    return velocity


def return_linear_virtual_velocity_gradient_2(points, data):
    num_points = np.shape(points)[0]
    velocity = np.ones((num_points,2,2))
    for i in range(num_points):
        velocity[i, :, :] = (i + 1) * np.array([[1, 2], [3, 4]])
    return velocity


def return_ones_but_two_for_yx_yy_virtual_velocity_gradient(points, data):
    num_points = np.shape(points)[0]
    velocity = np.ones((num_points,2,2))
    velocity[:,1,:] = 2
    
    return velocity


def poly_space_exp_time(coords, time):
    base = 1 + coords[:,0] + np.power(coords[:,1], 2)
    data = np.zeros((len(time), len(base)))
    for ti, t in enumerate(time):
        data[ti, :] = base * np.exp(t/5)
    return data


class TestFieldTimeInterpolator(MatcalUnitTest):

    def setUp(self):
        super().setUp(__file__)

    def test_interpolate_filed_points(self):
        n_points = 4
        n_times = 40
        test_fun = poly_space_exp_time
        np.random.seed(123456)
        coords = np.random.uniform(0, 1, [n_points,2])
        source_times = np.linspace(0, 1, n_times)
        goal_times = np.linspace(0, 1, int(n_times*1.5))
        surface  = {'all':np.arange(n_points)}
        source_z = test_fun(coords, source_times)
        goal_z = test_fun(coords, goal_times)

        source_data = {'X':coords[:,0], "Y":coords[:,1], "Z":source_z, 
            "time":source_times, 'surf':surface}
        source_data = convert_dictionary_to_field_data(source_data, ['X', "Y"], 
            node_set_name='surf')
        source_data.skeleton.subset_name = 'all'
        ref_data = convert_dictionary_to_data({'time':goal_times})

        qoi_ex = FieldTimeInterpolatorExtractor(source_data.skeleton, 'time')
        qoi = qoi_ex.calculate(source_data, ref_data, ['Z'])
        self.assert_close_arrays(goal_z, qoi['Z'])


class TestMeshlessSpaceInterpolator(MatcalUnitTest):

    def setUp(self):
        super().setUp(__file__)

    def test_single_time_interp(self):
        n_points = 10
        source_mult = 10
        n_times = 1
        test_fun  = poly_space_exp_time
        self._confirm_qoi(n_points, source_mult, n_times, test_fun)

    def test_multiple_time_interp(self):
        n_points = 10
        source_mult = 10
        n_times = 9
        test_fun  = poly_space_exp_time
        self._confirm_qoi(n_points, source_mult, n_times, test_fun)

    def _confirm_qoi(self, n_points, source_mult, n_times, test_fun):
        np.random.seed(123456)
        source_coords = np.random.uniform(0, 1, [n_points*source_mult, 2])
        target_corrds = np.random.uniform(0, 1, [n_points, 2])
        source_time = np.linspace(0,1, n_times)

        source_function_data = test_fun(source_coords, source_time)
        goal_function_data = test_fun(target_corrds, source_time)

        source_data = {"x":source_coords[:,0], 'y':source_coords[:,1], 
            "time":source_time, 'func':source_function_data}
        source_data = convert_dictionary_to_field_data(source_data, ['x','y'])
        
        qoi_ex = MeshlessSpaceInterpolatorExtractor(source_data.skeleton.spatial_coords, 
            target_corrds, 'time', 2, 2)
        qoi = qoi_ex.calculate(source_data, None, ['func'])
        self.assert_close_arrays(qoi['func'], goal_function_data)
        qoi_ex.clean_up()


class TestNewToOldRemapper(MatcalUnitTest):

    def setUp(self):
        super().setUp(__file__)
        self.mapper = NewToOldRemapper()
    
    def test_same_returns_sequencial(self):
        np.random.seed(123456)
        repeats = 10
        for i in range(repeats):
            n_pts = np.random.randint(2, 20)
            x = np.random.uniform(-10, 10, n_pts)
            y = np.random.uniform(-1, 1, n_pts)
            pts = np.array([x, y]).T
            node_map = self.mapper(pts, pts)
            goal = list(range(n_pts))
            self.assert_close_arrays(node_map, goal)
    
    def test_larger_new_array_raises_error(self):    
        np.random.seed(123456)
        n_pts = np.random.randint(6, 20)
        x = np.random.uniform(-10, 10, n_pts)
        y = np.random.uniform(-1, 1, n_pts)
        old_pts = np.array([x, y]).T
        x_new = np.random.uniform(-5, 5, 5)
        y_new = np.random.uniform(-5, 5, 5)
        new_pts = np.array([np.concatenate([x, x_new]), np.concatenate([y, y_new])])
        with self.assertRaises(NewToOldRemapper.DifferentPointSetError):
            node_map = self.mapper(new_pts, old_pts)
    
    def test_smaller_new_array_raises_error(self):
        x_old = [1,2,3]
        y_old = [-1,-1, 2]
        x_new = [1,2]
        y_new = [-1, -1]
        pts_old = np.array([x_old, y_old]).T
        pts_new = np.array([x_new, y_new]).T
        with self.assertRaises(NewToOldRemapper.DifferentPointSetError):
            node_map = self.mapper(pts_new, pts_old)
    
    def test_reversed_arrays_returns_decending(self):
        repeats = 10
        np.random.seed(123456)
        for i in range(repeats):
            n_pts = np.random.randint(2, 20)
            x = np.random.uniform(-10, 10, n_pts)
            y = np.random.uniform(-1, 1, n_pts)
            pts_old = np.array([x, y]).T
            pts_new = np.array([np.flip(x), np.flip(y)]).T
            node_map = self.mapper(pts_new, pts_old)
            goal = np.flip(np.array(list(range(n_pts))))
            self.assert_close_arrays(node_map, goal)

    def test_randomized_order(self):
        repeats = 10
        np.random.seed(123456)
        for i in range(repeats):
            n_pts = np.random.randint(2, 20)
            x = np.random.uniform(-10, 10, n_pts)
            y = np.random.uniform(-1, 1, n_pts)
            goal = np.array(list(range(n_pts)))
            np.random.shuffle(goal)
            pts_new = np.array([x, y]).T
            pts_old = pts_new[goal, :]
            node_map = self.mapper(pts_new, pts_old)
            self.assert_close_arrays(node_map, goal)

    def test_get_identity_mapping(self):
        repeats = 10
        np.random.seed(123456)
        for i in range(repeats):
            n_pts = np.random.randint(2, 20)
            x = np.random.uniform(-10, 10, n_pts)
            y = np.random.uniform(-1, 1, n_pts)
            goal = np.array(list(range(n_pts)))
            np.random.shuffle(goal)
            pts_new = np.array([x, y]).T
            pts_old = pts_new[goal, :]
            new_to_old = self.mapper(pts_new, pts_old)
            old_to_new = self.mapper(pts_old, pts_new)
            self.assert_close_arrays(pts_new, pts_new[new_to_old,:][old_to_new, :])
            self.assert_close_arrays(pts_old, pts_old[old_to_new, :][new_to_old,:])

    def test_raise_error_for_redundant_point(self):
        n_pts = 10
        np.random.seed(123456)
        x = np.random.uniform(-10, 10, n_pts)
        y = np.random.uniform(-1, 1, n_pts)
        pts_old = np.array([x, y]).T
        pts_new = np.copy(pts_old)
        pts_new[7,:] = pts_old[2,:]
        with self.assertRaises(NewToOldRemapper.NonuniqueMappingError):
            self.mapper(pts_new, pts_old)

    def test_raise_error_for_unmapped_point(self):
        n_pts = 10
        np.random.seed(123456)
        x = np.random.uniform(-10, 10, n_pts)
        y = np.random.uniform(-1, 1, n_pts)
        pts_old = np.array([x, y]).T
        pts_new = np.copy(pts_old)
        pts_new[7,:] += 100.0
        with self.assertRaises(NewToOldRemapper.NoMapFoundError):
            self.mapper(pts_new, pts_old)


class TestHWDPolynomialSurfaceExtractor(MatcalUnitTest):

    def setUp(self):
        super().setUp(__file__)

    def test_get_identical_weights_for_same_data(self):
        n_time = 3
        n_pts = 6
        raw_data = {'x':np.array([-1.1,-.1, -.9, 1.1, 1., .9]), 
            'y':np.array([1, 1.1, 1.2, 1, 1.1, 1.2]), 'time':np.array([1,2,3])}
        raw_data['temp'] = np.ones([n_time, n_pts])
        data = convert_dictionary_to_field_data(raw_data,['x', 'y'])
        topo = data.skeleton
        depth = 1
        poly_order = 2
        time_field = 'time'
        qoi_tool = HWDPolynomialSimulationSurfaceExtractor(topo, depth, poly_order, time_field)
        weights1 = qoi_tool.calculate(data, data, ['temp'])
        weights2 = qoi_tool.calculate(data, data, ['temp'])
        self.assert_close_dicts_or_data(weights1, weights2)

    def test_get_identical_weights_for_constant_data_different_times(self):
        n_pts = 6
        raw_data1 = {'x':np.array([-1.1,-.1, -.9, 1.1, 1., .9]), 
            'y':np.array([1, 1.1, 1.2, 1, 1.1, 1.2]), 'time':np.array([1,2,3])}
        raw_data1['temp'] = np.ones([3, n_pts])
        data1 = convert_dictionary_to_field_data(raw_data1,['x', 'y'])

        raw_data2 = {'x':np.array([-1.1,-.1, -.9, 1.1, 1., .9]), 
            'y':np.array([1, 1.1, 1.2, 1, 1.1, 1.2]), 'time':np.array([.5,1.5,2.5,3.5])}
        raw_data2['temp'] = np.ones([4, n_pts])
        data2 = convert_dictionary_to_field_data(raw_data2,['x', 'y'])

        depth = 1
        poly_order = 2
        time_field = 'time'
        qoi_tool = HWDPolynomialSimulationSurfaceExtractor(data1.skeleton, 
            depth, poly_order, time_field)
        weights1 = qoi_tool.calculate(data1, data1, ['temp'])
        weights2 = qoi_tool.calculate(data2, data1, ['temp'])
        self.assert_close_dicts_or_data(weights1, weights2)

    def _linear(self, xs, ys, times):
        value = np.zeros([len(times), len(xs)])
        for row_i, t in enumerate(times):
            for col_i, (x, y) in enumerate(zip(xs, ys)):
                value[row_i, col_i] = x + y + t  
        return value
    
    def test_get_identical_weights_for_linear_data_different_times(self):
        n_pts = 6
        raw_data1 = {'x':np.array([-1.1,-.1, -.9, 1.1, 1., .9]), 
            'y':np.array([1, 1.1, 1.2, 1, 1.1, 1.2]), 'time':np.array([1,2,3])}
        raw_data1['temp'] = self._linear(raw_data1['x'], raw_data1['y'], raw_data1['time'])
        data1 = convert_dictionary_to_field_data(raw_data1,['x', 'y'])

        raw_data2 = {'x':np.array([-1.1,-.1, -.9, 1.1, 1., .9]), 
            'y':np.array([1, 1.1, 1.2, 1, 1.1, 1.2]), 'time':np.array([.5,1.5,2.5,3.5])}
        raw_data2['temp'] = self._linear(raw_data2['x'], raw_data2['y'], raw_data2['time'])
        data2 = convert_dictionary_to_field_data(raw_data2,['x', 'y'])

        depth = 1
        poly_order = 2
        time_field = 'time'
        qoi_tool = HWDPolynomialSimulationSurfaceExtractor(data1.skeleton, depth, 
            poly_order, time_field)
        weights1 = qoi_tool.calculate(data1, data1, ['temp'])
        weights2 = qoi_tool.calculate(data2, data1, ['temp'])
        self.assert_close_dicts_or_data(weights1, weights2)

    def test_get_close_weights_for_linear_data_different_times_and_points(self):
        n_pts = 6
        raw_data1 = {'x':np.array([-1.1,-.1, -.9, 1.1, 1., .9]), 
            'y':np.array([1, 1.1, 1.2, 1, 1.1, 1.2]), 'time':np.array([1,2,3])}
        raw_data1['temp'] = self._linear(raw_data1['x'], raw_data1['y'], raw_data1['time'])
        data1 = convert_dictionary_to_field_data(raw_data1,['x', 'y'])

        raw_data2 = {'x':np.array([-1.2,-1., -.85, 1.05, 1.02, .93]), 
            'y':np.array([1.02, 1.08, 1.17, 1.01, 1.11, 1.21]), 'time':np.array([.5,1.5,2.5,3.5])}
        raw_data2['temp'] = self._linear(raw_data2['x'], raw_data2['y'], raw_data2['time'])
        data2 = convert_dictionary_to_field_data(raw_data2,['x', 'y'])

        depth = 1
        poly_order = 2
        time_field = 'time'
        qoi_tool = HWDPolynomialSimulationSurfaceExtractor(data1.skeleton, depth, 
            poly_order, time_field)
        weights1 = qoi_tool.calculate(data1, data1, ['temp'])
        weights2 = qoi_tool.calculate(data2, data1, ['temp'])
        self.assert_close_dicts_or_data(weights1, weights2)

    def test_experimental_form_only_uses_first_entry(self):
        n_pts = 6
        raw_data1 = {'x':np.array([-1.1,-.1, -.9, 1.1, 1., .9]), 
            'y':np.array([1, 1.1, 1.2, 1, 1.1, 1.2]), 'time':np.array([1,2,3])}
        raw_data1['temp'] = self._linear(raw_data1['x'], raw_data1['y'], raw_data1['time'])
        data1 = convert_dictionary_to_field_data(raw_data1,['x', 'y'])

        raw_data2 = {'x':np.array([-1.2,-1., -.85, 1.05, 1.02, .93]), 
            'y':np.array([1.02, 1.08, 1.17, 1.01, 1.11, 1.21]), 'time':np.array([.5,1.5,2.5,3.5])}
        raw_data2['temp'] = self._linear(2*raw_data2['x'], raw_data2['y'], 4*raw_data2['time'])
        data2 = convert_dictionary_to_field_data(raw_data2,['x', 'y'])

        depth = 1
        poly_order = 2
        time_field = 'time'
        sim_qoi_tool = HWDPolynomialSimulationSurfaceExtractor(data1.skeleton, 
            depth, poly_order, time_field)
        exp_qoi_tool = HWDExperimentSurfaceExtractor(sim_qoi_tool)
        weights1 = sim_qoi_tool.calculate(data1, data1, ['temp'])
        weights2 = exp_qoi_tool.calculate(data1, data2, ['temp'])
        self.assert_close_dicts_or_data(weights1, weights2)

    def test_colocation_exp_extractor_linear(self):
        n_pts_sim = 20
        n_pts_exp = 40
        np.random.seed(123456)
        raw_data_sim = {'x':np.random.uniform(-1, 1, n_pts_sim), 
            'y':np.random.uniform(-1, 1, n_pts_sim), 'time':np.array([.5,1.5,2.5,3.5])}
        raw_data_sim['temp'] = self._linear(raw_data_sim['x'], 
            raw_data_sim['y'], raw_data_sim['time'])
        data_sim = convert_dictionary_to_field_data(raw_data_sim,['x', 'y'])

        raw_data_exp = {'x':np.random.uniform(-1, 1, n_pts_exp), 
            'y':np.random.uniform(-1, 1, n_pts_exp), 'time':np.array([1,2,3])}
        raw_data_exp['temp'] = self._linear(raw_data_exp['x'], raw_data_exp['y'], 
            raw_data_exp['time'])
        data_exp = convert_dictionary_to_field_data(raw_data_exp,['x', 'y'])

        depth = 1
        poly_order = 2
        sim_qoi_tool = HWDPolynomialSimulationSurfaceExtractor(data_sim.skeleton, 
            depth, poly_order, "time")
        exp_qoi_tool = HWDColocatingExperimentSurfaceExtractor(sim_qoi_tool, 
            data_exp.skeleton.spatial_coords, data_sim.skeleton.spatial_coords)

        weights2 = exp_qoi_tool.calculate(data_exp, data_exp, ['temp'])
        weights1 = sim_qoi_tool.calculate(data_sim, data_exp, ['temp'])
        self.assert_close_dicts_or_data(weights1, weights2)

    def _trig(self, xs, ys, times):
        value = np.zeros([len(times), len(xs)])
        for row_i, t in enumerate(times):
            for col_i, (x, y) in enumerate(zip(xs, ys)):
                value[row_i, col_i] = np.sin(np.pi/2 * (x + y)) *( 1 + t)
        return value

    def test_colocation_exp_extractor_trig(self):
        n_pts_sim = 200
        n_pts_exp = 2* n_pts_sim
        np.random.seed(123456)
        raw_data_sim = {'x':np.random.uniform(-1, 1, n_pts_sim), 
            'y':np.random.uniform(-1, 1, n_pts_sim), 'time':np.array([.5,1.5,2.5,3.5])}
        T_sim = self._trig(raw_data_sim['x'], raw_data_sim['y'], raw_data_sim['time'])
        raw_data_sim['temp'] = T_sim
        data_sim = convert_dictionary_to_field_data(raw_data_sim,['x', 'y'])

        raw_data_exp = {'x':np.random.uniform(-1, 1, n_pts_exp), 
            'y':np.random.uniform(-1, 1, n_pts_exp), 'time':np.array([1,2,3])}
        T_exp = self._trig(raw_data_exp['x'], raw_data_exp['y'], raw_data_exp['time'])
        raw_data_exp['temp'] = T_exp
        data_exp = convert_dictionary_to_field_data(raw_data_exp,['x', 'y'])

        depth = 2
        poly_order = 10
        time_field = 'time'
        sim_qoi_tool = HWDPolynomialSimulationSurfaceExtractor(data_sim.skeleton, depth, 
            poly_order, time_field)
        exp_qoi_tool = HWDColocatingExperimentSurfaceExtractor(sim_qoi_tool, 
            data_exp.skeleton.spatial_coords, data_sim.skeleton.spatial_coords, 8, 1.5)
        weights_exp = exp_qoi_tool.calculate(data_exp, data_exp, ['temp'])
        weights_sim = sim_qoi_tool.calculate(data_sim, data_exp, ['temp'], False)
        self.assert_close_dicts_or_data(weights_sim, weights_exp, err_tol = 1e-4)

    def test_colocation_exp_extractor_trig_one_time(self):
        n_pts_sim = 200
        n_pts_exp = 2* n_pts_sim
        np.random.seed(123456)
        raw_data_sim = {'x':np.random.uniform(-1, 1, n_pts_sim), 
            'y':np.random.uniform(-1, 1, n_pts_sim), 'time':np.array([0])}
        T_sim = self._trig(raw_data_sim['x'], raw_data_sim['y'], raw_data_sim['time'])
        raw_data_sim['temp'] = T_sim
        data_sim = convert_dictionary_to_field_data(raw_data_sim,['x', 'y'])

        raw_data_exp = {'x':np.random.uniform(-1, 1, n_pts_exp), 
            'y':np.random.uniform(-1, 1, n_pts_exp), 'time':np.array([0])}
        T_exp = self._trig(raw_data_exp['x'], raw_data_exp['y'], raw_data_exp['time'])
        raw_data_exp['temp'] = T_exp
        data_exp = convert_dictionary_to_field_data(raw_data_exp,['x', 'y'])

        depth = 2
        poly_order = 10
        time_field = 'time'
        sim_qoi_tool = HWDPolynomialSimulationSurfaceExtractor(data_sim.skeleton, 
            depth, poly_order, time_field)
        exp_qoi_tool = HWDColocatingExperimentSurfaceExtractor(sim_qoi_tool,
            data_exp.skeleton.spatial_coords, data_sim.skeleton.spatial_coords, 8, 1.5)
        weights_exp = exp_qoi_tool.calculate(data_exp, data_exp, ['temp'])

        weights_sim = sim_qoi_tool.calculate(data_sim, data_exp, ['temp'], False)
        self.assert_close_dicts_or_data(weights_sim, weights_exp, err_tol = 1e-4)

    def make_one_element_cube_same_node_numbering(self):
        grid = {}
        grid['x'] = np.array([-1, 1, 1, -1, -1, 1, 1, -1]) 
        grid['y'] = np.array([-1, -1, 1, 1, -1, -1, 1, 1])
        grid['z'] = np.array([1, 1, 1, 1, -1, -1, -1, -1])
        grid['connectivity'] = np.array([[0, 1, 2, 3], [7, 6, 5, 4], 
            [1, 5, 6, 2], [2, 6, 7 , 3], [0, 3, 7 ,4], [0, 4, 5, 1]])
        grid['node_sets'] = {'front':np.array([0, 1, 2, 3])}
        return grid

    def _x_linear(self, xs, ys, times):
        value = np.zeros([len(times), len(xs)])
        for row_i, t in enumerate(times):
            for col_i, (x, y) in enumerate(zip(xs, ys)):
                value[row_i, col_i] = x + t  
        return value

    def test_extract_simulation_data_from_FEM_surface_source_same_time(self):
        n_pts = 6
        cloud_data_dict = {'x':np.array([-1.1,-.5, -.9, 1.1, 5., .9]), 
            'y':np.array([1, 1.1, -1.2, -1, 1.1, 1.2]), 'time':np.array([1,2,3])}
        cloud_data_dict['temp'] = self._x_linear(cloud_data_dict['x'], 
            cloud_data_dict['y'], cloud_data_dict['time'])
        cloud_data = convert_dictionary_to_field_data(cloud_data_dict,['x', 'y'])

        mesh_dict = self.make_one_element_cube_same_node_numbering()
        mesh_dict['time'] =  cloud_data_dict['time']
        mesh_dict['temp'] = self._x_linear(mesh_dict['x'], mesh_dict['y'], mesh_dict['time'])
        mesh_data = convert_dictionary_to_field_data(mesh_dict, ['x', 'y', 'z'], 
            'connectivity', 'node_sets')

        depth = 0
        poly_order = 1
        time_field = 'time'
        qoi_tool = HWDPolynomialSimulationSurfaceExtractor(cloud_data.skeleton, depth, 
            poly_order, time_field)
        qoi_tool.extract_cloud_from_mesh('front')
        weights_cloud = qoi_tool.calculate(cloud_data, cloud_data, ['temp'])
        weights_mesh_to_cloud = qoi_tool.calculate(mesh_data, cloud_data, ['temp'])
        self.assert_close_arrays(weights_cloud, weights_mesh_to_cloud)


def return_zeros_virtual_velocity(points, data):
    velocity = np.zeros((np.shape(points)[0],2))
    return velocity


def return_threes_virtual_velocity(points, data):
    velocity = np.ones((np.shape(points)[0],2))*3
    return velocity


class TestExternalVirtualPowerExtractor(MatcalUnitTest):

    def setUp(self):
        super().setUp(__file__)

    def test_return_zero_with_zero_velocity(self):
        n_times = 3
        exp_load_data_dict = {'load': np.ones(n_times), 'time':np.linspace(0,1,n_times), 
            'x':[0,0,1,1], 'y':[0,1,0,1]}
        exp_field_data = convert_dictionary_to_field_data(exp_load_data_dict, ["x","y"])
        self._assert_power_from_velocity_and_load(exp_field_data, np.zeros(n_times), 
            return_zeros_virtual_velocity)

    def _assert_power_from_velocity_and_load(self, exp_load_data, goal, 
        virtual_velocty_func = None):

        if virtual_velocty_func is None:
            used_vv_fun = _default_velocity_function
        else:
            used_vv_fun = virtual_velocty_func
        evpe = ExternalVirtualPowerExtractor("time", "load", used_vv_fun)
        fields_of_interest = None
        results = evpe.calculate(exp_load_data, exp_load_data, fields_of_interest)
        self.assert_close_arrays(results['virtual_power'], goal)

    def test_return_zero_with_zero_load(self):
        n_times = 3
        exp_load_data_dict = {'load': np.zeros(n_times), 'time':np.linspace(0,1,n_times), 
            'x':[0,0,1,1], 'y':[0,1,0,1]}
        exp_field_data = convert_dictionary_to_field_data(exp_load_data_dict, ["x","y"])
        self._assert_power_from_velocity_and_load(exp_field_data, np.zeros(n_times))

    def test_return_constants_with_uniform_load_and_vel(self):
        n_times = 3
        virtual_velocity = np.array([0,3])
        exp_load_data_dict = {'load': np.ones(n_times)*2, 'time':np.linspace(0,1,n_times), 
            'x':[0,0,1,1], 'y':[0,1,0,1]}
        exp_field_data = convert_dictionary_to_field_data(exp_load_data_dict, ["x","y"])
        goal = np.ones(n_times) * 6
        self._assert_power_from_velocity_and_load(exp_field_data, goal, 
            return_threes_virtual_velocity)

    def test_return_linear_power_increase(self):
        n_times = 3
        virtual_velocity = np.array([2,1])
        exp_load_data_dict = {'load': np.linspace(0,2,n_times), 'time':np.linspace(0,1,n_times), 
            'x':[0,0,1,1], 'y':[0,1,0,1]}
        exp_field_data = convert_dictionary_to_field_data(exp_load_data_dict, ["x","y"])
        self._assert_power_from_velocity_and_load(exp_field_data, exp_load_data_dict['load'])


class TestFieldDataFlattener(MatcalUnitTest):

    def setUp(self):
        super().setUp(__file__)

    def test_flatten_two_dim_array(self):
        np.random.seed(123456)
        fields = ['a','b']
        data_dict = {'a':np.ones([5,4]), 'b':np.random.random([5,4])}
        fake_qoi = convert_dictionary_to_data(data_dict)
        extractor = FlattenFieldDataExtractor()
        qoi = extractor.calculate(fake_qoi, None, fields)
        for f in fields: 
            self.assert_close_arrays(qoi[f], data_dict[f].flatten())


def return_zeros_virtual_velocity_gradient(points, data):
    velocity = np.zeros((np.shape(points)[0],2,2))
    return velocity


def return_threes_virtual_velocity_gradient(points, data):
    velocity = np.ones((np.shape(points)[0],2,2))*3
    return velocity

def return_linear_virtual_velocity_gradient(points, data):
    num_points = np.shape(points)[0]
    velocity = np.ones((num_points,2,2))
    for point_i in range(num_points):
        velocity[point_i, :, :] = -(point_i+1)*np.ones((2,2))*3
    
    return velocity


def return_linear_virtual_velocity_gradient_2(points, data):
    num_points = np.shape(points)[0]
    velocity = np.ones((num_points,2,2))
    for i in range(num_points):
        velocity[i, :, :] = (i + 1) * np.array([[1, 2], [3, 4]])
    return velocity


def return_ones_but_two_for_yx_yy_virtual_velocity_gradient(points, data):
    num_points = np.shape(points)[0]
    velocity = np.ones((num_points,2,2))
    velocity[:,1,:] = 2
    
    return velocity


def poly_space_exp_time(coords, time):
    base = 1 + coords[:,0] + np.power(coords[:,1], 2)
    data = np.zeros((len(time), len(base)))
    for ti, t in enumerate(time):
        data[ti, :] = base * np.exp(t/5)
    return data


class TestInternalVirtualPowerExtractor(MatcalUnitTest):
    

    class InternalVirtualPowerExtractorSpy(InternalVirtualPowerExtractor):

        def spy_stress_fields(self):
            return self._stress_fields

        def spy_interpolation(self, interp_source, reference_source, ref_field):
            return self._interpolate_evaluation_data_to_projection_data(interp_source, 
                reference_source, ref_field)

        def spy_calculate_volume_scaled_velocity_gradient(self, working_data ):
            return self._calculate_volume_scaled_velocity_gradient(working_data)

    def setUp(self):
        super().setUp(__file__)
        
    def test_init(self):
        ivpe = InternalVirtualPowerExtractor("time", _default_velocity_gradient_function)

    def test_multiply_constant_areas_and_velocity_correctly(self):
        ntimes = 3
        ones = [[1,1,1,1], [1,1,1,1], [1,1,1,1]]
        data_dict = {"time":np.linspace(0,1,ntimes), "x":[0, 0, 1, 1], "y":[0,1,0,1], 
                     "element_area":ones, 
                     "element_thickness":ones,
                     "first_pk_stress_xx":ones, 
                     "first_pk_stress_yy":ones,
                     "first_pk_stress_xy":ones,
                     "first_pk_stress_yx":ones, 
                     "centroid_x":[[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1]],
                     "centroid_y":[[0,1,0,1], [0,1,0,1], [0,1,0,1]]}
        
        def ones_velocity_grad_func(points, field_data):
            velocity_grad = np.ones((points.shape[0],2,2))
            return velocity_grad
        
        data = convert_dictionary_to_field_data(data_dict, ["x", "y"])
        goal = np.ones([4, 2, 2])
        ivpe = self.InternalVirtualPowerExtractorSpy("time", ones_velocity_grad_func)
        scaled_vel_grad = ivpe.spy_calculate_volume_scaled_velocity_gradient(data)

        self.assert_close_arrays(goal, ivpe.spy_calculate_volume_scaled_velocity_gradient(data))

    def test_multiply_constant_volumes_and_velocity_correctly(self):
        ntimes = 3
        ones = [[1,1,1,1], [1,1,1,1], [1,1,1,1]]
        data_dict = {"time":np.linspace(0,1,ntimes), "x":[0, 0, 1, 1], "y":[0,1,0,1], 
                     "volume":ones, 
                     "first_pk_stress_xx":ones, 
                     "first_pk_stress_yy":ones,
                     "first_pk_stress_xy":ones,
                     "first_pk_stress_yx":ones, 
                     "centroid_x":[[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1]],
                     "centroid_y":[[0,1,0,1], [0,1,0,1], [0,1,0,1]]}
        
        def ones_velocity_grad_func(points, field_data):
            velocity_grad = np.ones((points.shape[0],2,2))
            return velocity_grad
        data = convert_dictionary_to_field_data(data_dict, ["x", "y"])
        goal = np.ones([4, 2, 2])
        ivpe = self.InternalVirtualPowerExtractorSpy("time", ones_velocity_grad_func)
        scaled_vel_grad = ivpe.spy_calculate_volume_scaled_velocity_gradient(data)
        self.assert_close_arrays(goal, ivpe.spy_calculate_volume_scaled_velocity_gradient(data))

    def test_multiply_linear_areas_and_constant_velocity_correctly(self):

        n_times = 3
        n_cells=4
        ones = np.ones((n_times, n_cells))
        data_dict = {"time":np.linspace(0,1,n_times), "x":[0, 0, 1, 1], "y":[0,1,0,1], 
                     "element_thickness":ones, 
                     "element_area":ones,
                     "first_pk_stress_xx":ones, 
                     "first_pk_stress_yy":ones,
                     "first_pk_stress_xy":ones,
                     "first_pk_stress_yx":ones, 
                     "centroid_x":[[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1]],
                     "centroid_y":[[0,1,0,1], [0,1,0,1], [0,1,0,1]]}
        data = convert_dictionary_to_field_data(data_dict, ["x", "y"])
        data["element_area"] = -1 * (np.arange(n_cells) + 1)

        goal = np.ones([n_cells, 2, 2])
        for i in range(n_cells):
            goal[i, :, :] = -(i + 1) * np.ones([2, 2])*3

        ivpe = self.InternalVirtualPowerExtractorSpy("time", 
            return_threes_virtual_velocity_gradient)
        self.assert_close_arrays(goal, ivpe.spy_calculate_volume_scaled_velocity_gradient(data))

    def test_multiply_linear_areas_and_linear_velocity_correctly(self):
        n_times = 3
        n_cells=4
        ones = np.ones((n_times, n_cells))
        data_dict = {"time":np.linspace(0,1,n_times), "x":[0, 0, 1, 1], "y":[0,1,0,1], 
                     "element_thickness":ones, 
                     "element_area":ones,
                     "first_pk_stress_xx":ones, 
                     "first_pk_stress_yy":ones,
                     "first_pk_stress_xy":ones,
                     "first_pk_stress_yx":ones, 
                     "centroid_x":[[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1]],
                     "centroid_y":[[0,1,0,1], [0,1,0,1], [0,1,0,1]]}
        data = convert_dictionary_to_field_data(data_dict, ["x", "y"])
        data["element_area"] = 1 * (np.arange(n_cells) + 1)
        goal = np.ones([n_cells, 2, 2])
        for i in range(n_cells):
            goal[i, :, :] = -((i + 1) ** 2) * np.ones([2, 2])*3

        ivpe = self.InternalVirtualPowerExtractorSpy("time", 
            return_linear_virtual_velocity_gradient)
        vol_scale_velo_grad = ivpe.spy_calculate_volume_scaled_velocity_gradient(data)
        self.assert_close_arrays(goal, vol_scale_velo_grad)

    def test_return_zeros_from_zero_gradients(self):
        ntimes = 3
        ones = [[1,1,1,1], [1,1,1,1], [1,1,1,1]]
        data_dict = {"time":np.linspace(0,1,ntimes), "x":[0, 0, 1, 1], "y":[0,1,0,1], 
                     "volume":ones, 
                     "first_pk_stress_xx":ones, 
                     "first_pk_stress_yy":ones,
                     "first_pk_stress_xy":ones,
                     "first_pk_stress_yx":ones, 
                     "centroid_x":[[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1]],
                     "centroid_y":[[0,1,0,1], [0,1,0,1], [0,1,0,1]]}
        data = convert_dictionary_to_field_data(data_dict, ["x", "y"])

        goal = {}
        goal['time'] = np.linspace(0,1,ntimes)
        goal['virtual_power'] = np.zeros([ntimes])

        self._confirm_result(data, goal, 
            virtual_velocity_gradient=return_zeros_virtual_velocity_gradient)

    def _confirm_result(self, data, goal, virtual_velocity_gradient):
        fields_of_interest = None
        ivpe = InternalVirtualPowerExtractor("time", virtual_velocity_gradient)
        result = ivpe.calculate(data, goal, fields_of_interest)
        self.assert_close_arrays(result['virtual_power'], goal['virtual_power'])

    def test_return_constant_from_constant_vel_gad_and_stress(self):
        n_cells = 6
        n_times = 6
        ones = np.ones((n_times, n_cells))
        x_locs = [0, 0, 1, 1, 2, 2]
        y_locs = [0, 1, 0, 1, 0, 1]
        data_dict = {"time":np.linspace(0,1,n_times), "x":x_locs, "y":y_locs, 
                     "element_thickness":ones, 
                     "element_area":ones,
                     "first_pk_stress_xx":ones, 
                     "first_pk_stress_yy":ones,
                     "first_pk_stress_xy":ones,
                     "first_pk_stress_yx":ones, 
                     "centroid_x":[x_locs]*n_times,
                     "centroid_y":[y_locs]*n_times}
        data = convert_dictionary_to_field_data(data_dict, ["x", "y"])
        goal = {}
        goal["virtual_power"] = np.ones([n_times]) * 4 * n_cells * 3
        goal["time"] = data["time"]
        self._confirm_result(data, goal, return_threes_virtual_velocity_gradient)

    def test_return_linear_result_from_linear_stress(self):
        n_cells = 6
        x_locs = [0, 0, 1, 1, 2, 2]
        y_locs = [0, 1, 0, 1, 0, 1]
        times = np.array([0, .75, 1.5, 2.0])
        n_times = len(times)
        ones = np.ones((n_times, n_cells))

        data_dict = {"time":times, "x":x_locs, "y":y_locs, 
                     "element_thickness":ones, 
                     "element_area":ones,
                     "first_pk_stress_xx":np.outer((np.ones(n_cells)) * 0, times).T, 
                     "first_pk_stress_yy":np.outer((np.ones(n_cells)) * 1, times).T,
                     "first_pk_stress_xy":np.outer((np.ones(n_cells)) * 2, times).T,
                     "first_pk_stress_yx":np.outer((np.ones(n_cells)) * 3, times).T, 
                     "centroid_x":[x_locs]*n_times,
                     "centroid_y":[y_locs]*n_times}
        data = convert_dictionary_to_field_data(data_dict, ["x", "y"])

        time = np.linspace(0, 2, 6)
        goal = {}
        goal['time'] = time
        goal['virtual_power'] = time * n_cells * (0 * 1 + 1 * 2 + 2 * 1 + 3 * 2)

        self._confirm_result(data, goal, return_ones_but_two_for_yx_yy_virtual_velocity_gradient)

    def test_return_quadratic_result_from_linear_stress_and_linear_velocity(self):
        n_cells = 6
        x_locs = [0, 0, 1, 1, 2, 2]
        y_locs = [0, 1, 0, 1, 0, 1]
        times = np.array([0, .75, 1.5, 2.0])
        n_times = len(times)
        ones = np.ones((n_times, n_cells))
        data_dict = {"time":times, "x":x_locs, "y":y_locs, 
                     "element_thickness":ones, 
                     "element_area":ones,
                     "first_pk_stress_xx":np.outer((np.arange(n_cells)+1) * 1, times).T, 
                     "first_pk_stress_yy":np.outer((np.arange(n_cells)+1) * 4, times).T,
                     "first_pk_stress_xy":np.outer((np.arange(n_cells)+1) * 2, times).T,
                     "first_pk_stress_yx":np.outer((np.arange(n_cells)+1) * 3, times).T, 
                     "centroid_x":[x_locs]*n_times,
                     "centroid_y":[y_locs]*n_times}
        data = convert_dictionary_to_field_data(data_dict, ["x", "y"])
        time = np.linspace(0, 2, 6)
 
        goal = {}
        goal['virtual_power'] = (time * (1 + 4 + 9 + 16) * 
            np.sum(np.power(np.arange(n_cells) + 1, 2)))
        goal['time'] = time
        self._confirm_result(data, goal, return_linear_virtual_velocity_gradient_2)

    def test_confirm_default_projection_fields(self):
        ivpe = self.InternalVirtualPowerExtractorSpy("time", _default_velocity_gradient_function)
        field_list = ['first_pk_stress_xx', 'first_pk_stress_yy', 'first_pk_stress_xy',
                      'first_pk_stress_yx']
        for field_goal in field_list:
            self.assertIn(field_goal, ivpe.spy_stress_fields())

    def test_confirm_constant_interpolation(self):
        n_times=6
        goal = np.ones([n_times])
        exp_load_data = {'time': np.linspace(0, 2.5, n_times)}
        fields_of_interest = None
        ivpe = self.InternalVirtualPowerExtractorSpy("time", _default_velocity_gradient_function)
        interp = ivpe.spy_interpolation(exp_load_data["time"], np.linspace(0, 2.5, 3), np.ones([3]))
        self.assert_close_arrays(interp, goal)

    def test_confirm_linear_interpolation(self):

        n_times=6
        goal = 2*np.linspace(0,2,n_times)
        exp_load_data = {'time': np.linspace(0, 2, n_times)}
        fields_of_interest = None
        ivpe = self.InternalVirtualPowerExtractorSpy("time", _default_velocity_gradient_function)
        interp = ivpe.spy_interpolation(exp_load_data["time"], 
            np.linspace(0, 2.5, 3), 2*np.linspace(0,2.5,3))
        self.assert_close_arrays(interp, goal)
 
    def test_confirm_thickness_scaling_with_nonunit_thickness(self):
        n_cells = 6
        n_times = 10
        cell_areas = np.ones(n_cells)

        virtual_velocity_gradient = np.ones([n_cells, 2, 2]) * 2
        scale = 7
        ivpe = self.InternalVirtualPowerExtractorSpy("time", 
            return_threes_virtual_velocity_gradient)
        ivpe_scaled = self.InternalVirtualPowerExtractorSpy("time", 
            return_threes_virtual_velocity_gradient)

        ntimes = 3
        ones = [[1,1,1,1], [1,1,1,1], [1,1,1,1]]
        data_dict = {"time":np.linspace(0,1,ntimes), "x":[0, 0, 1, 1], "y":[0,1,0,1], 
                     "element_area":ones, 
                     "element_thickness":ones,
                     "first_pk_stress_xx":ones, 
                     "first_pk_stress_yy":ones,
                     "first_pk_stress_xy":ones,
                     "first_pk_stress_yx":ones, 
                     "centroid_x":[[-1, -1, 1, 1], [-1, -1, 1, 1], [-1, -1, 1, 1]],
                     "centroid_y":[[-1,1,-1,1], [-1,1,-1,1], [-1,1,-1,1]]}
       
        data = convert_dictionary_to_field_data(data_dict, ["x", "y"])
        import copy
        scaled_data = copy.deepcopy(data)
        scaled_data["element_thickness"] = scaled_data["element_thickness"]*7

        ratio = np.divide(ivpe_scaled.spy_calculate_volume_scaled_velocity_gradient(scaled_data), 
                          ivpe.spy_calculate_volume_scaled_velocity_gradient(data))
        self.assertAlmostEqual(np.sum(ratio/scale), np.size(ratio))

    def _create_field_goal(self, idx, n_cells, n_times, time):
        goal = np.zeros([n_times, n_cells])
        stencil = time * (idx)
        for i in range(n_cells):
            goal[:, i] = stencil
        return goal

