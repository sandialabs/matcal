import numpy as np

from matcal.core.data import DataCollection
from matcal.core.tests.MatcalUnitTest import MatcalUnitTest
from matcal.core.tests.unit.test_data import CommonDataUnitTests

from matcal.full_field.data import FieldData, convert_dictionary_to_field_data


class TestFieldData(CommonDataUnitTests.CommonTests):

    _data_class = FieldData

    def test_add_and_return_node_positions(self):
        n_pts = 10
        x = np.random.uniform(0, 10, n_pts)
        y = np.random.uniform(0, 10, n_pts)
        pts = np.array([x,y]).T
        self._data.set_spatial_coords(pts)
        self.assert_close_arrays(pts, self._data.spatial_coords)

    def test_add_and_return_node_connectivity(self):
        con = np.random.randint(0, 20, [6,4])
        self._data.set_connectivity(con)
        self.assert_close_arrays(con, self._data.connectivity)

    def test_data_collection_dumps(self):
        n_pts = 10
        data_dict = {}
        data_dict["t"] = np.linspace(0, 10, self._len)
        data_dict["T"] = np.linspace(0.25, 11, self._len)
        data_dict["U"] = np.random.uniform(0, 1, (self._len, n_pts))
        data_dict["x"] = np.random.uniform(0, 10, n_pts)
        data_dict["y"] = np.random.uniform(0, 10, n_pts)
        data = convert_dictionary_to_field_data(data_dict, coordinate_names=['x','y'])

        dc = DataCollection("test", data)

        dump = dc.dumps(ignore_point_data=True)
        self.assertTrue("U" not in list(dump["matcal_default_state"][0].keys()))
        dump = dc.dumps()
        self.assertTrue("U" in list(dump["matcal_default_state"][0].keys()))

    def test_num_dimensions_return_based_on_coord_dimensions(self):
        n_pts = 10
        data_dict = {}
        data_dict["t"] = np.linspace(0, 10, self._len)
        data_dict["T"] = np.linspace(0.25, 11, self._len)
        data_dict["U"] = np.random.uniform(0, 1, (self._len, n_pts))
        data_dict["x"] = np.random.uniform(0, 10, n_pts)
        data_dict["y"] = np.random.uniform(0, 10, n_pts)
        data_dict['z'] = np.random.uniform(0, 10, n_pts)
        data_2d = convert_dictionary_to_field_data(data_dict, coordinate_names=['x','y'])
        self.assertEqual(data_2d.num_dimensions, 2)
        data_3d = convert_dictionary_to_field_data(data_dict, coordinate_names=['x','y', 'z'])
        self.assertEqual(data_3d.num_dimensions, 3)
        
    def test_num_dimensions_raise_error_if_no_coordinates_provided(self):
        n_pts = 10
        test_dict = {"a":np.linspace(0, 10, n_pts)}
        data = convert_dictionary_to_field_data(test_dict)
        with self.assertRaises(RuntimeError):
            data.num_dimensions
            
    def test_num_nodes_returns_0_if_undefined(self):
        n_pts = 10
        test_dict = {"a":np.linspace(0, 10, n_pts)}
        data = convert_dictionary_to_field_data(test_dict)
        self.assertEqual(data.num_nodes, 0)
        
    def test_num_nodes_is_number_of_coords(self):
        n_pts = 10
        data_dict = {}
        data_dict["t"] = np.linspace(0, 10, self._len)
        data_dict["T"] = np.linspace(0.25, 11, self._len)
        data_dict["U"] = np.random.uniform(0, 1, (self._len, n_pts))
        data_dict["x"] = np.random.uniform(0, 10, n_pts)
        data_dict["y"] = np.random.uniform(0, 10, n_pts)
        data_dict['z'] = np.random.uniform(0, 10, n_pts)
        data_2d = convert_dictionary_to_field_data(data_dict, coordinate_names=['x','y'])
        self.assertEqual(data_2d.num_nodes, n_pts)
        data_3d = convert_dictionary_to_field_data(data_dict, coordinate_names=['x','y', 'z'])
        self.assertEqual(data_3d.num_nodes, n_pts)

    def test_num_elements_returns_0_if_undef(self):
        n_pts = 10
        test_dict = {"a":np.linspace(0, 10, n_pts)}
        data = convert_dictionary_to_field_data(test_dict)
        self.assertEqual(data.num_nodes, 0)
        
    def test_num_elements_returns_length_of_connectivity(self):
        n_pts = 10
        data_dict = {}
        data_dict["t"] = np.linspace(0, 10, self._len)
        data_dict["T"] = np.linspace(0.25, 11, self._len)
        data_dict["U"] = np.random.uniform(0, 1, (self._len, n_pts))
        data_dict["x"] = np.random.uniform(0, 10, n_pts)
        data_dict["y"] = np.random.uniform(0, 10, n_pts)
        data_dict['z'] = np.random.uniform(0, 10, n_pts)
        data_dict['con'] = np.array([[0, 1, 2], 
                                     [1,2, 3],
                                     [3, 4, 5],
                                     [4, 5, 6],
                                     [6, 7, 8]])
        n_ele = data_dict['con'].shape[0]
        data_2d = convert_dictionary_to_field_data(data_dict, coordinate_names=['x','y'], connectivity_name='con')
        self.assertEqual(data_2d.num_elements, n_ele)
        data_3d = convert_dictionary_to_field_data(data_dict, coordinate_names=['x','y', 'z'], connectivity_name='con')
        self.assertEqual(data_3d.num_elements, n_ele)

class TestConvertFullFieldDictToData(MatcalUnitTest):

    def setUp(self):
        super().setUp(__file__)

    def test_convert_full_field_dict_to_data_only(self):
        n_pts = 5
        n_time = 10
        x = np.linspace(0, 1, n_pts)
        y = np.linspace(0, 1, n_pts)
        X, Y = np.meshgrid(x, y)
        x = X.flatten()
        y= Y.flatten()
        time = np.linspace(0, 4, n_time)
        Temp = np.random.uniform(0, 1, [n_pts, n_time]).T
        flux = np.random.uniform(10, 20, n_time)
        data_dict = {'time':time, "temp": Temp, 'flux':flux}
        goal_data_dict = {'time':time, "temp": Temp, 'flux':flux}

        data = convert_dictionary_to_field_data(data_dict)
        topology_info = data.skeleton

        for key in list(goal_data_dict.keys()):
            self.assert_close_arrays(data_dict[key], data[key])
        
        self.assertIsNone(topology_info.spatial_coords)
        self.assertIsNone(topology_info.connectivity)

    def test_convert_full_field_dict_to_data_and_location(self):
        n_pts = 5
        n_time = 10
        x = np.linspace(0, 1, n_pts)
        y = np.linspace(0, 1, n_pts)
        X, Y = np.meshgrid(x, y)
        x = X.flatten()
        y= Y.flatten()
        time = np.linspace(0, 4, n_time)
        Temp = np.random.uniform(0, 1, [n_pts, n_time]).T
        flux = np.random.uniform(10, 20, n_time)
        data_dict = {'time':time, 'x':x, 'y':y, "temp": Temp, 'flux':flux}
        goal_data_dict = {'time':time, "temp": Temp, 'flux':flux}

        data = convert_dictionary_to_field_data(data_dict, ['x', 'y'])
        topology_info = data.skeleton
        for key in list(goal_data_dict.keys()):
            self.assert_close_arrays(data_dict[key], data[key])
        
        self.assert_close_arrays(topology_info.spatial_coords[:,0], x)
        self.assert_close_arrays(topology_info.spatial_coords[:,1], y)
        self.assertIsNone(topology_info.connectivity)

    def test_convert_with_node_set(self):
        n_pts = 5
        n_time = 10
        side_points = [0,3,4]
        side_name = "myside"
        x = np.linspace(0, 1, n_pts)
        y = np.linspace(0, 1, n_pts)
        X, Y = np.meshgrid(x, y)
        x = X.flatten()
        y= Y.flatten()
        time = np.linspace(0, 4, n_time)
        Temp = np.random.uniform(0, 1, [n_pts, n_time]).T
        flux = np.random.uniform(10, 20, n_time)
        data_dict = {'time':time, 'x':x, 'y':y, "temp": Temp, 'flux':flux, 'node_sets':{side_name:side_points}}
        goal_data_dict = {'time':time, "temp": Temp, 'flux':flux}

        data = convert_dictionary_to_field_data(data_dict, ['x', 'y'], node_set_name='node_sets')
        topology_info = data.skeleton
        for key in list(goal_data_dict.keys()):
            self.assert_close_arrays(data_dict[key], data[key])
        
        self.assert_close_arrays(topology_info.spatial_coords[:,0], x)
        self.assert_close_arrays(topology_info.spatial_coords[:,1], y)
        self.assert_close_arrays(topology_info.surfaces[side_name], side_points)     

    def test_convert_full_field_dict_to_data(self):
        n_pts = 5
        n_time = 10
        side_points = [0,3,4]
        side_name = "myside"
        x = np.linspace(0, 1, n_pts)
        y = np.linspace(0, 1, n_pts)
        X, Y = np.meshgrid(x, y)
        x = X.flatten()
        y= Y.flatten()
        connectivity = np.array([list(range(0,n_pts-1)),list(range(1, n_pts))]).T
        time = np.linspace(0, 4, n_time)
        Temp = np.random.uniform(0, 1, [n_pts, n_time]).T
        flux = np.random.uniform(10, 20, n_time)
        data_dict = {'time':time, 'x':x, 'y':y, "temp": Temp, 'flux':flux, 'connectivity':connectivity, 'node_sets':{side_name:side_points}}
        goal_data_dict = {'time':time, "temp": Temp, 'flux':flux}

        data = convert_dictionary_to_field_data(data_dict, ['x', 'y'], 'connectivity', 'node_sets')
        topology_info = data.skeleton
        for key in list(goal_data_dict.keys()):
            self.assert_close_arrays(data_dict[key], data[key])
        
        self.assert_close_arrays(topology_info.spatial_coords[:,0], x)
        self.assert_close_arrays(topology_info.spatial_coords[:,1], y)
        self.assert_close_arrays(topology_info.connectivity, connectivity)
        self.assert_close_arrays(topology_info.surfaces[side_name], side_points)     
