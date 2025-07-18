from matcal.full_field.data_importer import ImportedTwoDimensionalMesh
import numpy as np
import os

from matcal.full_field.TwoDimensionalFieldGrid import GridAxis, MeshSkeleton
from matcal.full_field.TwoDimensionalFieldGrid import TwoDimensionalFieldGrid, \
    ContainingElementIdentifier, ElementBins, ElementIdentifierBase
from matcal.core.tests.MatcalUnitTest import MatcalUnitTest


class TestMeshSkeleton(MatcalUnitTest):
    def setUp(self) -> None:
        super().setUp(__file__)
    
    def test_init(self):
        points = None
        connectivity = None
        mesh = MeshSkeleton(points, connectivity)
        mesh2 = MeshSkeleton()

    def test_recall_points(self):
        points = np.array([[1,2],[0,0]])
        mesh = MeshSkeleton(points)
        self.assert_close_arrays(points, mesh.spatial_coords)

    def test_recall_connectivity(self):
        con = np.array([[0,1,2,3],[3,2,4,5]])
        mesh = MeshSkeleton(connectivity=con)
        self.assert_close_arrays(con, mesh.connectivity)

    def test_recall_surface_node_list(self):
        mesh = MeshSkeleton()
        nodes = np.array([0,1,2,3,4])
        one_surface = {"surface_name": nodes}
        mesh.add_node_sets(**one_surface)
        self.assert_close_arrays(nodes, mesh.surfaces['surface_name'])

    def test_add_multiple_surfaces(self):
        mesh = MeshSkeleton()
        s = {}
        for i in range(10):
            name = f"surf_{i}"
            nodes = np.random.randint(0, 100, np.random.randint(5, 20))
            s[name]= nodes
        mesh.add_node_sets(**s)
        for key, value in s.items():
            self.assert_close_arrays(value, mesh.surfaces[key])
    
    def test_format_for_serial(self):
        ms = MeshSkeleton()
        n_pts = 20
        n_dim = 3
        n_ele = 5
        ele_size = 8
        points = np.random.uniform(0,10, [n_pts, n_dim])
        connectivity = np.random.randint(0, n_pts, [n_ele, ele_size])
        name = 'my_name'
        node_sets = {"a":np.random.randint(0, n_pts, n_pts//3).tolist(), 'b':np.random.randint(0, n_pts, n_pts//2).tolist()}
        ms = MeshSkeleton(points, connectivity)
        ms.add_node_sets(**node_sets)
        ms.subset_name = name
        serial = ms.serialize()
        goal_dict = {"spatial_coords":points.tolist(), "connectivity":connectivity.tolist(), "subset_name":name, "surfaces":node_sets}
        goal_surf = goal_dict.pop('surfaces')
        test_surf = serial.pop('surfaces')
        self.assert_close_dicts_or_data(goal_surf, test_surf)
        self.assert_close_dicts_or_data(goal_dict, serial)



class TestTwoDimensionalFieldGrid(MatcalUnitTest):
    def setUp(self) -> None:
        super().setUp(__file__)
        self.x = GridAxis(0, 6, 4)
        self.y = GridAxis(0, 4, 3)
        self.thickness = 1
        self.tdfg = TwoDimensionalFieldGrid(self.x, self.y)

    def test_return_node_positions(self):
        nodes = self.tdfg.node_positions
        goal_nodes = np.array([[0., 0.],
                               [2., 0.],
                               [4., 0.],
                               [6., 0.],
                               [0., 2.],
                               [2., 2.],
                               [4., 2.],
                               [6., 2.],
                               [0., 4.],
                               [2., 4.],
                               [4., 4.],
                               [6., 4.]])
        self.assertIsInstance(nodes, np.ndarray)
        self.assertTrue(np.allclose(nodes, goal_nodes))

    def test_return_cell_connectivity(self):
        connection = self.tdfg.cell_connectivity
        goal_connection = np.array([[0, 1, 5, 4],
                                    [1, 2, 6, 5],
                                    [2, 3, 7, 6],
                                    [4, 5, 9, 8],
                                    [5, 6, 10, 9],
                                    [6, 7, 11, 10]]
                                   )
        self.assertIsInstance(connection, np.ndarray)
        self.assertTrue(np.array_equal(connection, goal_connection))

    def test_correct_cell_count(self):
        self.assertEqual(self.tdfg.cell_count, 6)

    def test_correct_node_count(self):
        self.assertEqual(self.tdfg.node_count, 12)

    def test_return_correct_connectivity(self):
        first = self.tdfg.get_cell_nodes(0)
        last = self.tdfg.get_cell_nodes(5)
        middle = self.tdfg.get_cell_nodes(3)

        self.assertTrue(np.array_equal(first, [0, 1, 5, 4]))
        self.assertTrue(np.array_equal(last, [6, 7, 11, 10]))
        self.assertTrue(np.array_equal(middle, [4, 5, 9, 8]))

    def test_return_correct_containing_cell_from_center_positions(self):
        positions = [[1, 1], [5, 1], [5, 3], [3, 3]]
        goal_cells = [0, 2, 5, 4]
        for idx in range(len(goal_cells)):
            pos = positions[idx]
            cell = self.tdfg.get_containing_cell(pos)
            self.assertEqual(cell, goal_cells[idx])

    def test_return_array_of_containing_cells(self):
        positions = np.array([[1, 1], [5, 1], [5, 3], [3, 3]])
        goal_cells = [0, 2, 5, 4]
        cells = self.tdfg.get_containing_cell(positions)
        self.assertTrue(np.array_equal(cells, goal_cells))

    def test_return_node_locations_for_element(self):
        node_position = self.tdfg.get_cell_node_locations(0)
        goal = np.array([[0, 0], [2, 0], [2, 2], [0, 2]])
        self.assertTrue(np.allclose(node_position, goal))

    def test_uniform_cell_size_array(self):
        y = GridAxis(0, 8, 3)
        thickness = 1
        tdfg = TwoDimensionalFieldGrid(self.x, y)
        goal_sizes = np.ones([3, 6])
        goal_sizes[0, :] = 2
        goal_sizes[1, :] = 4
        goal_sizes[2, :] = 3

        self.assertTrue(np.allclose(tdfg.cell_sizes, goal_sizes))

    def test_get_cell_centers(self):
        centers = self.tdfg.get_cell_centers()
        goal_centers = np.array([[1, 1],
                                 [3, 1],
                                 [5, 1],
                                 [1, 3],
                                 [3, 3],
                                 [5, 3]
                                 ])
        self.assertTrue(np.allclose(centers, goal_centers))

    def test_get_cell_areas(self):
        areas = self.tdfg.get_cell_areas()
        goal_areas = np.ones(6) * 4
        self.assert_close_arrays(areas, goal_areas)

    def test_get_grid_corners(self):
        corners = self.tdfg.get_grid_corners()
        goal = np.array([[0, 0], [6, 0], [6, 4], [0, 4]])
        self.assert_close_arrays(corners, goal)


class TestImportedTwoDimensionalGrid(MatcalUnitTest):
    class ImportedTwoDimensionalMeshSpy(ImportedTwoDimensionalMesh):

        def spy_longest_length(self):
            return self._longest_length

        def spy_bin_length(self):
            return self._calculate_bin_count_and_sizes()[1]

        def spy_bin_count(self):
            return self._calculate_bin_count_and_sizes()[0]

    def setUp(self):
        super().setUp(__file__)
        path = self.get_current_files_path(__file__)
        self._square_mesh_file = "/".join([path, "input_files", "flat_rectangle.json"])
        self._5x5_mesh_file = "/".join([path, "input_files", "flat_rectangle_5x5.json"])
        self._uneven_mesh_file = "/".join([path, "input_files", "flat_rectangle_uneven.json"])
        self._thickness = .01
        self.square_grid = self.ImportedTwoDimensionalMeshSpy(self._square_mesh_file)
        self.node_locations_id_order = np.array([[.5, .5],
                                                 [0, .5],
                                                 [0, 0],
                                                 [.5, 0],
                                                 [-.5, .5],
                                                 [-.5, 0],
                                                 [0, -.5],
                                                 [.5, -.5],
                                                 [-.5, -.5]])
        self.connectivity_ids = np.array(
            [[1, 2, 3, 4], [2, 5, 6, 3], [4, 3, 7, 8], [3, 6, 9, 7]]) - 1  # account for exo

        self.uneven_grid = self.ImportedTwoDimensionalMeshSpy(self._uneven_mesh_file)
        self.fine_grid = self.ImportedTwoDimensionalMeshSpy(self._5x5_mesh_file)


    def test_get_connectivity(self):
        self.assert_close_arrays(self.connectivity_ids, self.square_grid.cell_connectivity)

    def test_get_node_positions(self):
        self.assert_close_arrays(self.node_locations_id_order, self.square_grid.node_positions)

    def test_number_of_notes_and_cells(self):
        self.assertEqual(self.square_grid.node_count, 9)
        self.assertEqual(self.square_grid.cell_count, 4)

    def test_get_cell_nodes(self):
        self.assert_close_arrays(self.square_grid.get_cell_nodes(1), [1, 4, 5, 2])
        self.assert_close_arrays(self.square_grid.get_cell_nodes([3, 0]), [[2, 5, 8, 6], [0, 1, 2, 3]])

    def test_get_cell_node_locations(self):
        for ele_id in range(4):
            self._confirm_element_node_locations(ele_id)

    def test_cell_areas_get_same_for_all_because_regular(self):
        goal_area = .25 * np.ones(self.square_grid.cell_count)
        self.assert_close_arrays(goal_area, self.square_grid.get_cell_areas())
        L = .571429
        M = .2857142
        S = .1428579
        goal_uneven = np.array([S*S, S*M, S*L, M*S, M*M, M*L, L*S, L*M, L*L])

        self.assert_close_arrays(goal_uneven,self.uneven_grid.get_cell_areas(), atol=1e-6 )

    def test_get_containing_cell(self):
        test_point = np.array([[.1, .1]])
        goal_cell = [0]
        self.assert_close_arrays(goal_cell, self.square_grid.get_containing_cell(test_point))

    def test_get_bin_size(self):
        goal_square = [1, 1]
        goal_uneven = [1, 1]
        goal_fine = [1 / 3, 1 / 3]
        self.assert_close_arrays(goal_square, self.square_grid.spy_bin_length())
        self.assert_close_arrays(goal_uneven, self.uneven_grid.spy_bin_length())
        self.assert_close_arrays(goal_fine, self.fine_grid.spy_bin_length())

    def test_get_bin_count(self):
        goal_square = [1, 1]
        goal_uneven = [1, 1]
        goal_fine = [3, 3]
        self.assert_close_arrays(goal_square, self.square_grid.spy_bin_count())
        self.assert_close_arrays(goal_uneven, self.uneven_grid.spy_bin_count())
        self.assert_close_arrays(goal_fine, self.fine_grid.spy_bin_count())

    def test_get_longest_edge(self):
        goal_square = np.sqrt(2) * .5
        goal_uneven = np.sqrt(2 * (.0714286 + .5) ** 2)
        goal_fine = np.sqrt(2) * .2
        self.assertAlmostEqual(goal_square, self.square_grid.spy_longest_length())
        self.assertAlmostEqual(goal_uneven, self.uneven_grid.spy_longest_length())
        self.assertAlmostEqual(goal_fine, self.fine_grid.spy_longest_length())

    def test_get_cell_centers(self):
        goal = np.array([[.25, .25], [-.25, .25], [.25, -.25], [-.25, -.25]])
        results =self.square_grid.get_cell_centers()
        self.assert_close_arrays(goal, results)

    def test_return_node_array(self):
        N = 36
        goal = np.zeros(N, dtype=int)
        for i in range(N):
            goal[i] = i
        self.assert_close_arrays(goal, self.fine_grid.node_list)

    def test_return_cell_array(self):
        N = 25
        goal = np.zeros(N, dtype=int)
        for i in range(N):
            goal[i] = i
        self.assert_close_arrays(goal, self.fine_grid.cell_list)

    def _confirm_element_node_locations(self, ele_id):
        nodes = self.connectivity_ids[ele_id]
        locations = self.node_locations_id_order[nodes, :]
        self.assert_close_arrays(self.square_grid.get_cell_node_locations(ele_id), locations)


class TestPointInElement(MatcalUnitTest):

    def setUp(self):
        super().setUp(__file__)

    class ceiSpy(ContainingElementIdentifier):

        def spy_distance_vector(self, *args):
            return self._calculate_distance_vector(*args)

        def spy_coordinate_value(self, location):
            return self._find_coordinates(location)

        def spy_matrix(self):
            return self._basis_matrix.todense()

    def test_form_distances(self):
        test_loc = np.array([1, 1])
        potential_elements = [0]
        connectivity = np.array([[0, 1, 2, 3]])
        node_locations = np.array([[0, 0], [2, 0], [2, 2], [0, 2]])
        goal = np.array([[1, 1], [-1, -1]])
        cei = self.ceiSpy( connectivity[potential_elements, :], node_locations)
        result = cei.spy_distance_vector(test_loc)
        self.assert_close_arrays(goal, result)

    def test_from_distances_multiple_elements(self):
        test_loc = np.array([1, 1])
        potential_elements = [0, 1]
        connectivity = np.array([[0, 1, 2, 3], [0, 4, 5, 1]])
        node_locations = np.array([[0, 0], [2, 0], [2, 2], [0, 2], [0, -1], [2, -1]])
        goal = np.array([[[1, 1], [-1, -1]], [[1, 1], [-1, 2]]])
        cei = self.ceiSpy(connectivity[potential_elements, :], node_locations)
        result = cei.spy_distance_vector(test_loc)
        self.assert_close_arrays(goal, result)

    def test_make_basis_matrix(self):
        connectivity = np.array([[0, 1, 2, 3], [0, 4, 5, 1]])
        node_locations = np.array([[0, 0], [2, 0], [2, 2], [0, 2], [0, -1], [2, -1]])
        cei = self.ceiSpy(connectivity, node_locations)
        goal = np.zeros([8, 8])
        goal[0:2, 0:2] = np.array([[2, 0], [0, 2]])
        goal[2:4, 2:4] = np.array([[0, -2], [-2, 0]])
        goal[4:6, 4:6] = np.array([[0, 2], [-1, 0]])
        goal[6:8, 6:8] = np.array([[-2, 0], [0, 1]])
        self.assert_close_arrays(goal, cei.spy_matrix())


    def test_get_basis_coordinates_rectangular(self):
        test_loc = np.array([1, 1])
        connectivity = np.array([[0, 1, 2, 3]])
        node_locations = np.array([[0, 0], [2, 0], [2, 2], [0, 2]])
        cei = self.ceiSpy(connectivity, node_locations)

        result = cei.spy_coordinate_value(test_loc)
        goal = np.array([[.5, .5], [.5, .5]]).flatten()
        self.assert_close_arrays(goal, result)


    def test_get_basis_coordinates_two_elements(self):
        connectivity = np.array([[0, 1, 2, 3], [0, 4, 5, 1]])
        node_locations = np.array([[0, 0], [2, 0], [2, 2], [0, 2], [0, -1], [2, -1]])
        cei = self.ceiSpy(connectivity, node_locations)
        test_loc = np.array([1, 1])
        result = cei.spy_coordinate_value(test_loc)
        goal = np.array([[.5, .5], [.5, .5], [-1 , .5], [.5, 2]]).flatten()
        self.assert_close_arrays(goal, result)

    def test_return_containing_element(self):
        connectivity = np.array([[0, 1, 2, 3], [0, 4, 5, 1]])
        node_locations = np.array([[0, 0], [2, 0], [2, 2], [0, 2], [0, -1], [2, -1]])
        cei = self.ceiSpy(connectivity, node_locations)
        self.assertEqual(cei.find_containing_element(np.array([1,1])), 0)
        self.assertEqual(cei.find_containing_element(np.array([1,-.5])), 1)
        self.assertEqual(cei.find_containing_element(np.array([0,0])), 0)
        self.assertIsNone(cei.find_containing_element(np.array([19,10])))


class TestElementBins(MatcalUnitTest):

    def setUp(self):
        super().setUp(__file__)

    def test_init(self):
        ElementBins(2,2,1,1,-1,-2)

    def test_bin_elements(self):
        eb = self._small_bin_setup()
        element_centers = np.array([[.1,.1]])
        connectivity = np.array([[0,1,2,3]])
        node_positions = np.array([[0,0],[1.,0],[1.,1],[0,1]])
        eb.bin(element_centers, connectivity, node_positions)
        bin_id = 7
        binned_elements = eb.get_elements(bin_id)
        self.assert_close_arrays(binned_elements, [0])
        self.assert_close_arrays(eb.get_elements(1), [])

    def test_edge_correction_x(self):
        eb = self._small_bin_setup()
        element_centers = np.array([[2.0,-.5]])
        connectivity = np.array([[0,1,2,3]])
        node_positions = np.array([[0,0],[1.,0],[1.,1],[0,1]])
        eb.bin(element_centers, connectivity, node_positions)
        bin_id = 5
        binned_elements = eb.get_elements(bin_id)
        self.assert_close_arrays(binned_elements, [0])
        self.assert_close_arrays(eb.get_elements(1), [])
    
    def test_edge_correction_y(self):
        eb = self._small_bin_setup()
        element_centers = np.array([[1.5,1]])
        connectivity = np.array([[0,1,2,3]])
        node_positions = np.array([[0,0],[1.,0],[1.,1],[0,1]])
        eb.bin(element_centers, connectivity, node_positions)
        bin_id = 8
        binned_elements = eb.get_elements(bin_id)
        self.assert_close_arrays(binned_elements, [0])
        self.assert_close_arrays(eb.get_elements(1), [])
    
    def test_edge_correction_x_and_y(self):
        eb = self._small_bin_setup()
        element_centers = np.array([[2,1]])
        connectivity = np.array([[0,1,2,3]])
        node_positions = np.array([[0,0],[1.,0],[1.,1],[0,1]])
        eb.bin(element_centers, connectivity, node_positions)
        bin_id = 8
        binned_elements = eb.get_elements(bin_id)
        self.assert_close_arrays(binned_elements, [0])
        self.assert_close_arrays(eb.get_elements(1), [])

    def _small_bin_setup(self):
        n = 3
        lx = 1
        ly = 1
        px = -1
        py = -2
        eb = ElementBins(n, n, lx, ly, px, py)
        return eb

    def _simple_2x2_bin(self):
        n = 2
        lx = 1
        ly = 1
        px = 0
        py = 0
        eb = ElementBins(n, n, lx, ly, px, py)
        return eb
    def test_throw_error_if_binned_a_second_time(self):
        eb = self._small_bin_setup()
        element_centers =np.ones([1,2])
        connectivity = np.array([[0,1,2,3]])
        node_positions = np.array([[0,0],[1.,0],[1.,1],[0,1]])
        eb.bin(element_centers, connectivity, node_positions)
        with self.assertRaises(ElementBins.RebinningError):
            eb.bin(element_centers, connectivity, node_positions)

    def test_get_element_lookup_post_binning_in_empty_and_full_bins(self):
        eb = self._2x2_bin()
        for i in range(4):
            self.assertIsInstance(eb._element_lookups[i], ElementIdentifierBase )

    def _2x2_bin(self):
        eb = self._simple_2x2_bin()
        element_centers = np.array([[.1, .1], [1.1, .1], [.1, 1.1]])
        connectivity = np.array([[0, 1, 2, 3], [1, 4, 5, 2], [3, 2, 6, 7]])
        nodes = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [2, 0], [2, 1], [1, 2], [0, 2]])
        eb.bin(element_centers, connectivity, nodes)
        return eb

    def test_point_lookup(self):
        eb = self._2x2_bin()
        location = np.array([.5,.5])
        goal_element = 0
        self.assertEqual(eb.find_containing_element(location), goal_element)
        self.assertEqual(eb.find_containing_element(np.array([1.2, .1])), 1)
        self.assertEqual(eb.find_containing_element(np.array([.2, 1.5])), 2)

    def test_return_neg_one_if_no_elements(self):
        eb = self._2x2_bin()
        self.assertEqual(eb.find_containing_element([1.5, 1.5]), -1)