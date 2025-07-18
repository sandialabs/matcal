import os
import numpy as np

from matcal.core.tests.MatcalUnitTest import MatcalUnitTest
from matcal.core.mesh_modifications import get_mesh_composer, get_mesh_decomposer

from matcal.full_field.data import convert_dictionary_to_field_data
from matcal.full_field.data_importer import (FieldSeriesData)

from matcal.exodus.mesh_modifications import (_element_lookup, extract_exodus_mesh, 
                                        SurfaceNotFoundError, 
                                        copy_mesh_and_store_data)
from matcal.exodus.tests.utilities import test_support_files_dir


def _build_ref_mesh(mesh_name, block_num=1):
    journal_str = f"""
    brick x 1 y 1 z 0.1
    vol all size 0.1
    mesh vol all
    block {block_num} vol all
    block {block_num} element type HEX8
    block {block_num} name "my_block"
    export mesh "{mesh_name}"
    """
    from matcal.sandia.tests.utilities import run_cubit_with_commands
    results = run_cubit_with_commands(journal_str.split("\n"))


def _create_field_data(t, point_locs, field_name, global_var=False):
    data = np.zeros((len(t), len(point_locs[:,0])))
    for index, time in enumerate(t):
        data[index, :] = (time*point_locs[:,0]+point_locs[:,1]/(time+0.1)+
                            point_locs[:,2]*time**2)
    data_dict = {"time":t, field_name:data}
    
    if global_var:
        gvar_vals = []
        for index, time in enumerate(data_dict["time"]):
            gvar_vals.append(np.linalg.norm(data_dict[field_name][index,:]))
        data_dict[field_name+"_gvar"] = gvar_vals
    field_data = convert_dictionary_to_field_data(data_dict)
    field_data.set_spatial_coords(point_locs)
    return field_data


class TestCopyMeshAndStoreNodeData(MatcalUnitTest):

    def setUp(self):
        super().setUp(__file__)
        self._mesh_name = os.path.join(test_support_files_dir, "test_mesh.g")

    def tearDown(self) -> None:
        super().tearDown()
    
    def test_copy_mesh_and_store_node_data(self):
        mesh_skeleton = extract_exodus_mesh(self._mesh_name)
        time = np.linspace(0,10,20)
        field_data = _create_field_data(time, mesh_skeleton.spatial_coords, 
                                          "val")
        mesh_with_data_filename = "test_result.e"

        copy_mesh_and_store_data(self._mesh_name, mesh_with_data_filename, 
                                      field_data, ["val"])
        
        mesh_data = FieldSeriesData(mesh_with_data_filename)
        for field in mesh_data.field_names:
            self.assert_close_arrays(field_data[field], mesh_data[field])

    def test_copy_mesh_and_store_node_data_then_store_new_data(self):
        mesh_skeleton = extract_exodus_mesh(self._mesh_name)
        time = np.linspace(0,10,20)
        field_data = _create_field_data(time, mesh_skeleton.spatial_coords, 
                                          "val")
        mesh_with_data_filename = "test_result.e"

        copy_mesh_and_store_data(self._mesh_name, mesh_with_data_filename, 
                                      field_data, ["val"])
        
        time = np.linspace(0,15,13)
        field_data_new = _create_field_data(time, mesh_skeleton.spatial_coords, 
                                          "val")
        mesh_with_new_data_filename = "test_result_new.e"

        copy_mesh_and_store_data(mesh_with_data_filename, 
                                      mesh_with_new_data_filename, 
                                      field_data_new, ["val"])

        mesh_data = FieldSeriesData(mesh_with_new_data_filename)
        for field in mesh_data.field_names:
            self.assert_close_arrays(field_data_new[field], mesh_data[field])

    def test_copy_mesh_and_store_node_and_global_data(self):
        mesh_skeleton = extract_exodus_mesh(self._mesh_name)
        time = np.linspace(0,10,20)
        field_data = _create_field_data(time, mesh_skeleton.spatial_coords, 
                                          "val", global_var=True)
        mesh_with_data_filename = "test_result.e"

        copy_mesh_and_store_data(self._mesh_name, mesh_with_data_filename, 
                                      field_data, ["val", "val_gvar", "time"])
        
        mesh_data = FieldSeriesData(mesh_with_data_filename)
        for field in mesh_data.field_names:
            self.assert_close_arrays(field_data[field], mesh_data[field])
        for field in field_data.field_names:
            self.assertTrue(field in mesh_data.field_names)

    def test_copy_mesh_and_store_node_and_global_data_from_field_data(self):
        mesh_skeleton = extract_exodus_mesh(self._mesh_name)
        time = np.linspace(0,10,20)
        field_data = _create_field_data(time, mesh_skeleton.spatial_coords, 
                                          "val", global_var=True)
        field_data.set_connectivity(mesh_skeleton.connectivity)
        mesh_with_data_filename = "test_result.e"

        copy_mesh_and_store_data(field_data, mesh_with_data_filename, 
                                      field_data, ["val", "val_gvar", "time"])
        
        mesh_data = FieldSeriesData(mesh_with_data_filename)
        for field in mesh_data.field_names:
            self.assert_close_arrays(field_data[field], mesh_data[field], show_on_fail=True)
        for field in field_data.field_names:
            self.assertTrue(field in mesh_data.field_names)


    def test_element_lookup(self):
        goals = {"TRI3":(2,3), "QUAD4":(2,4), "TET4":(3, 4), "HEX8": (3, 8)}
        for goal_name, (n_dim, n_node_per_ele) in goals.items():
            self.assertEqual(goal_name, _element_lookup(n_dim, n_node_per_ele))

    def test_raise_error_bad_element_lookup(self):
        with self.assertRaises(RuntimeError):
            _element_lookup(1, 3)
        with self.assertRaises(RuntimeError):
            _element_lookup(2, 2)
        with self.assertRaises(RuntimeError):
            _element_lookup(3, 2)


class TestExtractMeshGeometry(MatcalUnitTest):

    def setUp(self):
        super().setUp(__file__)
        self.test_support_files_path = os.path.join(os.path.dirname(__file__), '..', 
                                                    "test_support_files")

    def tearDown(self) -> None:
        super().tearDown()

    def test_extract_square_points_and_connectivity(self):
        filename = os.path.join(self.test_support_files_path, "square.e")
        mesh = extract_exodus_mesh(filename)
        goal_points = np.array([[0., 0., 0.],
                            [0., 1., 0.],
                            [1., 0., 0.],
                            [1., 1., 0.],
                            [0., 0., 1.],
                            [0., 1., 1.],
                            [1., 0., 1.],
                            [1., 1., 1.]])
        goal_con = np.array([[0, 1, 3, 2, 4, 5, 7, 6]])
        self.assert_close_arrays(goal_points, mesh.spatial_coords)
        self.assert_close_arrays(goal_con, mesh.connectivity)

    def test_extract_surface_points_and_connectivity(self):
        filename = os.path.join(self.test_support_files_path,  "cube8ele.g")
        surface_name = 'femsurface'
        mesh = extract_exodus_mesh(filename, surface_name)
        goal_points = np.array([[-0.5, -0.5,  0.5],
                [-0.5,  0.,   0.5],
                [ 0.,  -0.5,  0.5],
                [ 0.,   0.,   0.5],
                [-0.5,  0.5,  0.5],
                [ 0.,   0.5,  0.5],
                [ 0.5, -0.5,  0.5],
                [ 0.5,  0.,   0.5],
                [ 0.5,  0.5,  0.5]])
        goal_con = np.array([[2, 6, 7, 3,],
                [3, 7, 8, 5],
                [0, 2, 3, 1],
                [1, 3, 5, 4]])
        self.assert_close_arrays(goal_points, mesh.spatial_coords)

    def test_extract_surface_points_and_connectivity_bad_surface_string(self):
        filename = os.path.join(self.test_support_files_path, "cube8ele.g")
        surface_name = 'no_surface'
        with self.assertRaises(SurfaceNotFoundError):
            mesh = extract_exodus_mesh(filename, surface_name)

    def test_extract_surface_points_and_connectivity_bad_surface_type(self):
        filename = os.path.join(self.test_support_files_path, "cube8ele.g")
        surface_name = 'no_surface'
        with self.assertRaises(TypeError):
            mesh = extract_exodus_mesh(filename, [])

    def test_extract_surface_points_and_connectivity_bad_int(self):
        filename = os.path.join(self.test_support_files_path,  "cube8ele.g")
        surface_name = -1
        with self.assertRaises(SurfaceNotFoundError):
            mesh = extract_exodus_mesh(filename, surface_name)
    
    def test_extract_surface_points_and_connectivity_good_int(self):
        filename = os.path.join(self.test_support_files_path,  "cube8ele.g")
        surface_name = 1
        mesh = extract_exodus_mesh(filename, surface_name)


class TestExodusMeshCompDecomp(MatcalUnitTest):

    def setUp(self):
        super().setUp(__file__)
        self.path = self.get_current_files_path(__file__)

    def test_init(self):
        mesh_composer= get_mesh_composer('mesh.g')()
        
    def test_compose_a_decomposed_mesh(self):
        n_cores = 10
        base_mesh_name = "flat.g"
        source_mesh_name = os.path.join(self.path, test_support_files_dir, base_mesh_name)
        decomposer = get_mesh_decomposer(base_mesh_name)()
        composer = get_mesh_composer(base_mesh_name)()
        decomposer.decompose_mesh(source_mesh_name, n_cores)
        composer.compose_mesh(base_mesh_name, n_cores)
        self.assert_file_exists(base_mesh_name)




