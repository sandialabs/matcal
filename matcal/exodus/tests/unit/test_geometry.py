
import abc
import numpy as np
import os

from matcal.core.tests.MatcalUnitTest import MatcalUnitTest

from matcal.exodus.geometry import (ExodusHexGeometryCreator, NodeRelabeler,
                                    ExodusShellGeometryCreator)
from matcal.exodus.library_importer import create_exodus_class_instance
from matcal.exodus.mesh_modifications import _ExodusFieldInterpPreprocessor
from matcal.full_field.TwoDimensionalFieldGrid import GridAxis
from matcal.full_field.TwoDimensionalFieldGrid import TwoDimensionalFieldGrid

from matcal.full_field.data import convert_dictionary_to_field_data

n_cells = 6
n_x = 6 * 8
n_y = 8 * 8
n_nodes = n_x * n_y
x_locs, y_locs = np.meshgrid(np.linspace(0,1,n_x), np.linspace(0,2,n_y))
x_locs = x_locs.flatten()
y_locs = y_locs.flatten()

times = np.array([0, .75, 1.5, 2.0])
n_times = len(times)
ones = np.ones((n_times, n_nodes))
data_dict = {"time":times, "x":x_locs, "y":y_locs, 
                "U":np.outer(x_locs, times).T,
                "V":np.outer(y_locs, times).T}
mock_exo_data_linear = convert_dictionary_to_field_data(data_dict, ["x", "y"])

import copy
mock_exo_data_zeros = copy.deepcopy(mock_exo_data_linear)
mock_exo_data_zeros["U"] = np.zeros((n_times, n_nodes))
mock_exo_data_zeros["V"] = np.zeros((n_times, n_nodes))



class TestExodusGeometryCreatorBase():
    def __init__():
        pass
    
    class CommonTests(MatcalUnitTest, abc.ABC):

        @abc.abstractproperty
        def _nodes_per_element(self):
            """"""
        @abc.abstractproperty
        def _goal_node_set_names(self):
            """"""

        @abc.abstractproperty
        def _goal_node_set_ids(self):
            """"""

        @abc.abstractmethod
        def _get_goal_node_set_node_arrays(self):
            """"""

        @abc.abstractmethod
        def test_node_set_generation(self):
            """"""

        def setUp(self):
            super().setUp(__file__)
            self.nx = 3
            self.ny = 4
            self.n_nodes = self.nx * self.ny
            x_axis = GridAxis(0, 1, self.nx)
            y_axis = GridAxis(0, 2, self.ny)
            thickness = .5
            self._grid = TwoDimensionalFieldGrid(x_axis, y_axis)
            self._template_dir = "./"
            self._mesh_filename = "test_meshfile.g"
            self._egc = self._make_single_state_creator()

        def _make_single_state_creator(self):
            geo_params = {"reference_mesh_grid":self._grid, "thickness":0.5}
            egc = self._geometry_creator_class(self._mesh_filename,  geo_params)
            return egc

        def _make_mesh(self, egc=None):
            if egc is None:
                self._egc.create_mesh()
            else:
                egc.create_mesh()

        def confirm_node_set_ids_and_names(self):
            e = self._open_made_mesh()
            node_set_ids = e.get_node_set_ids()
            node_set_names = e.get_node_set_names()
            e.close()

            names_okay = True
            for name in self._goal_node_set_names:
                names_okay = names_okay and name in node_set_names
            goal_ids = self._goal_node_set_ids
            ids_okay = True
            for id in goal_ids:
                ids_okay = ids_okay and id in node_set_ids
            return names_okay and ids_okay


        def confrim_number_of_elements(self, goal):
            e = self._open_made_mesh()
            connectivity, n_ele, ele_size = e.get_elem_connectivity(1)
            e.close()
            return goal == n_ele

        def confirm_nodes_per_element(self):
            e = self._open_made_mesh()
            connectivity, n_ele, ele_size = e.get_elem_connectivity(1)
            e.close()
            return self._nodes_per_element == ele_size

        def confirm_node_set_count(self):
            e = self._open_made_mesh()
            node_set_ids = e.get_node_set_ids()
            e.close()
            return len(node_set_ids) == len(self._goal_node_set_ids) and \
                len(node_set_ids) == len(self._goal_node_set_names)

        def confirm_node_set_nodes(self):
            e = self._open_made_mesh()
            goal = self._get_goal_node_set_node_arrays()
            nodes_okay = True
            for ns_id, nodes in goal.items():
                ns_nodes = e.get_node_set_nodes(int(ns_id))
                ns_nodes.sort()
                nodes_okay = nodes_okay and np.allclose(ns_nodes, nodes)
            e.close()
            return nodes_okay

        def _open_made_mesh(self):
            e = create_exodus_class_instance(self._mesh_filename, array_type='numpy', mode='r')
            return e

        def confirm_zero_displacements(self):
            e = self._open_made_mesh()
            goal_variables = ["U", "V"]
            var_names_okay = True
            e_names = e.get_node_variable_names()
            goal_value = np.zeros(self._egc._get_number_of_nodes())
            for gvar in goal_variables:
                var_names_okay = var_names_okay and (gvar in e_names)
                for t_idx in range(len(e.get_times())):
                    n_set_var_vals = e.get_node_variable_values(gvar, t_idx + 1)
                    self.assert_close_arrays(n_set_var_vals, goal_value)
            self._confirm_time_steps(e, mock_exo_data_linear['time'])
            e.close()
            self.assertTrue(var_names_okay)
            
        def confirm_linear_assignment(self):
            e = self._open_made_mesh()
            goal_variables = {}
            num_nodes = self._egc._get_number_of_nodes()
            num_node_layers = self._get_num_node_layers()
            num_nodes_per_layer = int(num_nodes/num_node_layers)

            X, Y, Z = e.get_coords()
            e_time = e.get_times()
            goal_variables["U"] = np.outer(X, e_time).T
            goal_variables["V"] = np.outer(Y, e_time).T
            var_names_okay = True
            e_names = e.get_node_variable_names()
            p_times = mock_exo_data_linear['time']
            self._confirm_time_steps(e, p_times)
            for gvar, goal in goal_variables.items():
                var_names_okay = var_names_okay and (gvar in e_names)
                for t_idx in range(len(e.get_times())):
                    test = e.get_node_variable_values(gvar, t_idx + 1)
                    self.assert_close_arrays(test, goal[t_idx,:])

            self.assertTrue(var_names_okay)
            e.close()

        def _confirm_time_steps(self, exo_obj, time_array):
            self.assert_close_arrays(exo_obj.get_times(), time_array)

        def test_make_an_exodus_file(self):
            self._make_mesh()
            self.assertTrue(os.path.exists(self._mesh_filename))

        def test_write_file_and_confirm_correct_element_count(self):
            self._make_mesh()
            self.assertTrue(self.confrim_number_of_elements(6))
            self.assertTrue(self.confirm_nodes_per_element())

        def test_write_file_and_confirm_node_set_assignment(self):
            self._make_mesh()
            self.assertTrue(self.confirm_node_set_count())
            self.assertTrue(self.confirm_node_set_ids_and_names())
            self.assertTrue(self.confirm_node_set_nodes())

        def test_assign_zeros_displacements(self):
            self._make_mesh()
            mesh_field_processor = _ExodusFieldInterpPreprocessor()
            mesh_field_processor.process("./", self._mesh_filename, mock_exo_data_zeros, ["U", "V"], 2, 2)
            self.confirm_zero_displacements()

        def test_assign_linear_displacements(self):
            self._make_mesh()
            mesh_field_processor = _ExodusFieldInterpPreprocessor()
            mesh_field_processor.process("./", self._mesh_filename, mock_exo_data_linear, ["U", "V"], 2, 2)
            self.confirm_linear_assignment()

            
class TestExodusHexGeometryCreator(TestExodusGeometryCreatorBase.CommonTests):
    _geometry_creator_class = ExodusHexGeometryCreator
    _nodes_per_element = 8
    _goal_node_set_names = ["front_node_set", "back_node_set",
                    "fixed_z_node_set"]
    _goal_node_set_ids = [100, 200, 300]

    def _get_goal_node_set_node_arrays(self):
        node_count = self._egc._reference_mesh_grid.node_count
        goal = {'100': np.array(list(range(0, node_count))) + 1,
                '200': np.array(list(range(node_count, 2 * node_count))) + 1,
                '300': np.array([12]) + 1}
        return goal

    def test_make_nodes_at_thickness(self):
        full_nodes = self._egc._make_nodes_for_exo_mesh()
        current_positions = self._grid.node_positions
        front = np.concatenate((current_positions, np.zeros([self._grid.node_count, 1])), axis=1)
        back = np.concatenate((current_positions, np.ones([self._grid.node_count, 1]) * -self._egc._thickness), axis=1)
        goal = np.concatenate((front, back), axis=0)
        self.assert_close_arrays(full_nodes, goal)

    def test_create_node_connectivity_converter(self):
        new_con = self._egc._update_connectivity(np.array([[1, 2, 3, 4]]))
        node_count = self._grid.node_count
        goal = np.array([[1, 2, 2 + node_count, 1 + node_count, 4, 3, 3 + node_count, 4 + node_count]])
        self.assert_close_arrays(new_con, goal)

    def test_multiple_connectivity_conversion(self):
        new_con = self._egc._update_connectivity(np.array([[1, 2, 3, 4], [7, 6, 11, 10]]))
        node_count = self._grid.node_count
        goal = np.array([[1, 2, 2 + node_count, 1 + node_count, 4, 3, 3 + node_count, 4 + node_count],
                        [7, 6, 6 + node_count, 7 + node_count, 10, 11, 11 + node_count, 10 + node_count]])
        self.assert_close_arrays(new_con, goal)

    def test_node_set_generation(self):
        n_nodes_sheet = self._grid.node_count
        goal = self._get_goal_node_set_node_arrays()
        n_sets = self._egc._create_node_set_node_arrays()

        for i_set, goal_nset in enumerate(goal.values()):
            self.assert_close_arrays(n_sets[i_set], goal_nset)
        
    def _get_num_node_layers(self):
        return 2


class TestExodusShellGeometryCreator(TestExodusGeometryCreatorBase.CommonTests):
    _geometry_creator_class = ExodusShellGeometryCreator
    _nodes_per_element = 4
    _goal_node_set_names = ["surface_node_set"]
    _goal_node_set_ids = [100]

    def _get_goal_node_set_node_arrays(self):
        node_count = self._egc._reference_mesh_grid.node_count
        goal = {'100': np.array(list(range(0, node_count))) + 1}

        return goal

    def test_node_set_generation(self):
        n_nodes_sheet = self._grid.node_count
        goal = self._get_goal_node_set_node_arrays()
        n_sets = self._egc._create_node_set_node_arrays()
        for i_set, goal_nset in enumerate(goal.values()):
            self.assert_close_arrays(n_sets[i_set], goal_nset)

    def _get_num_node_layers(self):
        return 1
    

class TestNodeRelabeler(MatcalUnitTest):

    def setUp(self):
        super().setUp(__file__)
        self.nr = NodeRelabeler()

    def _confrim_relabel(self, old_list, goal_list):
        new_nodes = self.nr.relabel(old_list)
        self.assert_close_arrays(new_nodes, goal_list)

    def test_empty_list_return_empty(self):
        new_nodes = self.nr.relabel([])
        self.assertTrue(new_nodes.size == 0)

    def test_ordered_sequencial_list_return_same(self):
        old_list = [0, 1, 2, 3]
        self._confrim_relabel(old_list, old_list)

    def test_unordered_sequencial_list_return_same(self):
        old_list = [3, 0, 2, 1]
        self._confrim_relabel(old_list, old_list)

    def test_ordered_missing_1_number_return_sequencial(self):
        old_list = [0, 2, 3, 4]
        goal_list = [0, 1, 2, 3]
        self._confrim_relabel(old_list, goal_list)

    def test_unordered_missing_1_number(self):
        old_list = [5, 3, 1, 2, 0]
        goal_list = [4, 3, 1, 2, 0]
        self._confrim_relabel(old_list, goal_list)

    def test_unordered_missing_2_numbers(self):
        old_list = [5, 3, 1, 0]
        goal_list = [3, 2, 1, 0]
        self._confrim_relabel(old_list, goal_list)

    def test_does_not_start_at_0(self):
        old_list = [1, 2, 3, 4]
        goal_list = [0, 1, 2, 3]
        self._confrim_relabel(old_list, goal_list)

    def test_seqencial_missing_numbers(self):
        old_list = [0, 3, 4, 5]
        goal_list = [0, 1, 2, 3]
        self._confrim_relabel(old_list, goal_list)
