import csv
import numpy as np

from matcal.core.data import Data
from matcal.core.serializer_wrapper import json_serializer
from matcal.core.tests.MatcalUnitTest import MatcalUnitTest

from matcal.full_field.data import convert_dictionary_to_field_data
from matcal.full_field.data_importer import (CSVFieldDataSeriesParser, 
                                             _JSONFullFieldParser,
                                             _create_series_data_array, 
                                             FieldSeriesData, 
                                             _import_full_field_data_from_json, 
                                             mesh_file_to_skeleton, 
                                             _get_number_of_points_and_frames)
from matcal.full_field.data_exporter import export_full_field_data_to_json
from matcal.full_field.TwoDimensionalFieldGrid import MeshSkeleton


def read_csv(filename, header=None):
    with open(filename, 'r') as csvfile:

        if header is None:
            c = csv.DictReader(csvfile)
            keys = c.fieldnames
        else:
            keys = header
            c = csv.reader(csvfile)
        csv_dict = {}
        for key in keys:
            csv_dict[key] = []

        for row in c:
            for idx, key in enumerate(keys):
                if header is None:
                    lookup = key
                else:
                    lookup = idx
                csv_dict[key].append(float(row[lookup]))
        for key in keys:
            csv_dict[key] = np.array(csv_dict[key])
    return csv_dict

class TestSeriesImporter(MatcalUnitTest):

    def setUp(self) -> None:
        super().setUp(__file__)
        self._small_global_data = self.get_current_files_path(__file__)+"/input_files/csv_global_data.csv"
        self._small_series_directory = self.get_current_files_path(__file__)+"/input_files/csv_data_series"
    
    def test_bad_init(self):
        with self.assertRaises(TypeError):
            data = FieldSeriesData(1)

        with self.assertRaises(TypeError):
            data = FieldSeriesData(self._small_global_data, 1, ['X','Y'])

        with self.assertRaises(TypeError):
            data = FieldSeriesData(self._small_global_data, self._small_series_directory, 1)


        with self.assertRaises(Data.TypeError):
            data = FieldSeriesData(self._small_global_data, self._small_series_directory, ["X", "Y"], 1)

        with self.assertRaises(TypeError):
            data = FieldSeriesData(self._small_global_data, self._small_series_directory, ["X", "Y"], file_type=1)

        with self.assertRaises(ValueError):
            data = FieldSeriesData(self._small_global_data, self._small_series_directory, ["X", "Y"], n_cores="a")

        with self.assertRaises(ValueError):
            data = FieldSeriesData(self._small_global_data, self._small_series_directory, ["X", "Y"], n_cores=0)

    def test_seriesdata_values(self):
        series_data = FieldSeriesData(self._small_global_data, self._small_series_directory, ['X','Y'])
        goal_U = np.array([[0,0,0,0],[1,2,3,4],[1,2,3,4]])
        goal_V = np.array([[0,0,0,0],[0,0,0,0],[1,1,1,1]])
        self.assert_close_arrays(series_data['U'], goal_U)
        self.assert_close_arrays(series_data['V'], goal_V)
        goal_time = [0,1,2]
        goal_disp = [0, .01, .02]
        goal_load = [0, 10, 21]
        self.assert_close_arrays(series_data['time'], goal_time)
        self.assert_close_arrays(series_data['displacement'], goal_disp)
        self.assert_close_arrays(series_data['load'], goal_load)
    
    def test_series_data_small_parallel(self):
        goal_np = 4
        goal_nt = 3
        data_gold = FieldSeriesData(self._small_global_data, self._small_series_directory, 
                                    position_names=['X', 'Y'])
        data = FieldSeriesData(self._small_global_data, self._small_series_directory, n_cores=2, 
                               position_names=['X', 'Y'])
        self.assertEqual(len(data["time"]), goal_nt)
        self.assertEqual(data["U"].shape[1], goal_np)
        self.assert_close_arrays(data["U"], data_gold['U'])
        self.assert_close_arrays(data["V"], data_gold['V'])

        self.assert_close_arrays(data["time"], data_gold['time'])
        self.assert_close_arrays(data["load"], data_gold['load'])
        
    def test_seriesdata_positions(self):
        series_data = FieldSeriesData(self._small_global_data, self._small_series_directory, ['X','Y'])

        pos = np.array([[0,0],[1,0],[1,1],[0,1]])
        self.assert_close_arrays(pos, series_data.spatial_coords)
        self.assertIsNone(series_data.connectivity)

    def test_get_points_small_parallel(self):
        goal_np = 4
        goal_nt = 3
        fi = CSVFieldDataSeriesParser(self._small_global_data, self._small_series_directory, n_cores=2)
        np, nt, ne = _get_number_of_points_and_frames(fi)
        self.assertEqual(np, goal_np)
        self.assertEqual(nt, goal_nt)
        self.assertEqual(fi.number_of_cores, 2)

    def test_get_points_small(self):
        goal_np = 4
        goal_nt = 3
        fi = CSVFieldDataSeriesParser(self._small_global_data, self._small_series_directory)
        np, nt, ne = _get_number_of_points_and_frames(fi)
        self.assertEqual(np, goal_np)
        self.assertEqual(nt, goal_nt)
    
    def test_create_series_data_array_check_globals(self):
        fi = CSVFieldDataSeriesParser(self._small_global_data, self._small_series_directory)
        np, nt, ne = _get_number_of_points_and_frames(fi)
        position_names = ['X', 'Y']
        series_data_array = _create_series_data_array(fi, position_names)
        goal_keys = ['time', 'displacement', 'load']
        goal_time = [0,1,2]
        goal_disp = [0, .01, .02]
        goal_load = [0, 10, 21]
        keys = series_data_array.field_names
        for gkey in goal_keys:
            self.assertIn(gkey, goal_keys)
        self.assertTrue(not ('file' in keys))
        self.assert_close_arrays(series_data_array['time'], goal_time)
        self.assert_close_arrays(series_data_array['displacement'], goal_disp)
        self.assert_close_arrays(series_data_array['load'], goal_load)

    def test_create_series_data_array_check_space_keys(self):
        fi = CSVFieldDataSeriesParser(self._small_global_data, self._small_series_directory)
        np, nt, ne = _get_number_of_points_and_frames(fi)
        position_names = ['X', 'Y']
        series_data_array = _create_series_data_array(fi, position_names)
        goal_keys = ['U', 'V']
        keys = series_data_array.field_names
        for gkey in goal_keys:
            self.assertIn(gkey, goal_keys)
        for pkey in position_names:
            self.assertTrue(not (pkey in keys))

    def test_create_series_data_array_check_values(self):
        fi = CSVFieldDataSeriesParser(self._small_global_data, self._small_series_directory)
        num_p, num_t, num_e = _get_number_of_points_and_frames(fi)
        position_names = ['X', 'Y']
        series_data_array = _create_series_data_array(fi, position_names)
        goal_U = np.array([[0,0,0,0],[1,2,3,4],[1,2,3,4]])
        goal_V = np.array([[0,0,0,0],[0,0,0,0],[1,1,1,1]])
        self.assert_close_arrays(series_data_array['U'], goal_U)
        self.assert_close_arrays(series_data_array['V'], goal_V)


class TestJSONMeshSkeletonImporter(MatcalUnitTest):

    def setUp(self):
        super().setUp(__file__)

    def test_load_mesh_skeleton_same_points(self):
        my_mesh_skele = MeshSkeleton()
        n_pts = 10
        my_mesh_skele.spatial_coords = np.random.uniform(0,10, [n_pts, 2])
        imp_mesh_skele = self._write_then_read(my_mesh_skele)
        self.assert_close_arrays(my_mesh_skele.spatial_coords, imp_mesh_skele.spatial_coords)

    def _write_then_read(self, my_mesh_skele, surface=None):
        my_filename = "mesh_skele.json"
        with open(my_filename, 'w') as f:
            json_serializer.dump(my_mesh_skele.serialize(), f)
        imp_mesh_skele = mesh_file_to_skeleton(my_filename, surface)
        return imp_mesh_skele
    
    def test_load_mesh_skeleton_same_connectivity(self):
        my_mesh_skele = MeshSkeleton()
        n_ele = 10
        ele_size = 4
        my_mesh_skele.connectivity = np.random.randint(0, 100, [n_ele, ele_size])
        imp_mesh_skele = self._write_then_read(my_mesh_skele)
        self.assert_close_arrays(my_mesh_skele.connectivity, imp_mesh_skele.connectivity)

    def test_load_mesh_skeleton_same_name(self):
        src = MeshSkeleton()
        src.subset_name = 'test_name'
        target = self._write_then_read(src)
        self.assertEqual(src.subset_name, target.subset_name)

    def test_load_mesh_skeleton_same_subsets(self):
        src = MeshSkeleton()
        src.add_node_sets(a = np.random.randint(0, 100, 20), cat = np.random.randint(0, 12, 4))
        target = self._write_then_read(src)
        self.assert_close_dicts_or_data(target.surfaces, src.surfaces)

    def test_load_mesh_skeleton_only_surface(self):
        n_points = 20
        n_dim = 3
        n_ele = 20
        ele_size = 4
        pts = np.random.uniform(0, 100, [n_points, n_dim])
        ele = np.random.randint(0, n_points, [n_ele, ele_size])
        s_set = np.arange(0, n_points, 3)
        subset = {'myset':s_set}
        src = MeshSkeleton(pts, ele)
        src.add_node_sets(**subset)
        target = self._write_then_read(src, 'myset')
        self.assert_close_arrays(target.spatial_coords, pts[s_set,:])
        self.assertEqual(target.subset_name, 'myset')
        self.assertIsNone(target.connectivity)
        self.assert_close_dicts_or_data(target.surfaces, {})


class TestJSONFieldDataImporter(MatcalUnitTest):

    def setUp(self):
        super().setUp(__file__)

    def _make_simple_mesh_with_info(self):
        n_time = 3
        n_loc = 4
        time = np.linspace(0, 1, n_time)
        T = np.random.uniform(0,1,[n_time, n_loc])
        ref_ff_data = {'T':T, 'time':time, 'x':np.array([0, 1, 1, 0]), 'y':np.array([0, 0, 1, 1]), 'con':[[0, 1, 2, 3]]}
        ref_ff_data = convert_dictionary_to_field_data(ref_ff_data, ['x', 'y'], 'con')
        return ref_ff_data
    
    def test_get_same_data_imported_as_exported(self):
        base_data = self._make_simple_mesh_with_info()
        target_filename = 'target.json'
        export_full_field_data_to_json(target_filename, base_data)
        read_data = _import_full_field_data_from_json(target_filename)
        self.assert_close_dicts_or_data(base_data, read_data)

    def test_get_same_mesh_skeleton_imported_as_exported(self):
        base_data = self._make_simple_mesh_with_info()
        target_filename = 'target.json'
        export_full_field_data_to_json(target_filename, base_data)
        read_data = _import_full_field_data_from_json(target_filename)
        bs = base_data.skeleton
        rs = read_data.skeleton
        self.assert_close_arrays(bs.spatial_coords, rs.spatial_coords)
        self.assert_close_arrays(bs.connectivity, rs.connectivity)
        self.assert_close_dicts_or_data(bs.surfaces, rs.surfaces)
        self.assertIsNone(rs.subset_name)

class TestJSONFieldDataParser(MatcalUnitTest):

    def setUp(self):
        super().setUp(__file__)

    def _make_random_field_data(self, n_time, n_points, n_ele, ele_size, field_var_names, global_var_names):
        time = np.linspace(0, np.random.uniform(1, 10), n_time)
        x = np.random.uniform(-1, 1, n_points)
        y = np.random.uniform(-1, 1, n_points)
        z = np.random.uniform(-1, 1, n_points)
        data_dict = {'time':time, 'x':x, 'y':y, 'z':z}
        data_dict['connectivity'] = np.random.randint(0,n_points, [n_ele, ele_size])
        for var in field_var_names:
            data_dict[var] = np.random.uniform(0, 100, [n_time, n_points])
        for var in global_var_names:
            data_dict[var] = np.random.uniform(0, 100, [n_time])
        return convert_dictionary_to_field_data(data_dict, ['x', 'y', 'z'], 'connectivity')
    
    def test_json_parser_init(self):
        n_time = 10
        n_pts = 5
        n_ele = 1
        ele_size = 5
        field_vars = ['A', 'T']
        global_vars = ['total_heat']
        self._init_random_parse(n_time, n_pts, n_ele, ele_size, field_vars, global_vars)

    def _init_random_parse(self, n_time, n_pts, n_ele, ele_size, field_vars, global_vars):
        ref_data = self._make_random_field_data(n_time, n_pts, n_ele, ele_size, field_vars, global_vars)
        ref_filename = 'source.json'
        export_full_field_data_to_json(ref_filename, ref_data)
        json_parser = _JSONFullFieldParser(ref_filename)
        return json_parser, ref_data

    def test_get_correct_frame_count(self):
        n_time = 10
        n_pts = 5
        n_ele = 1
        ele_size = 5
        field_vars = ['A', 'T']
        global_vars = ['total_heat']
        parser, ref_data = self._init_random_parse(n_time, n_pts, n_ele, ele_size, field_vars, global_vars)
        self.assertEqual(parser.number_of_frames, n_time)
        n_time = 123
        parser, ref_data = self._init_random_parse(n_time, n_pts, n_ele, ele_size, field_vars, global_vars)
        self.assertEqual(parser.number_of_frames, n_time)

    def test_get_correct_global_field_names(self):
        n_time = 10
        n_pts = 5
        n_ele = 1
        ele_size = 5
        field_vars = ['A', 'T']
        global_vars = ['total_heat']
        goal_global_vars = ['time'] + global_vars
        parser, ref_data = self._init_random_parse(n_time, n_pts, n_ele, ele_size, field_vars, global_vars)
        self.assertEqual(len(goal_global_vars), len(parser.global_field_names))
        for gv in global_vars:
            self.assertIn(gv, parser.global_field_names)

        global_vars = ['total_Mac', 'total_and', 'total_cheese']
        goal_global_vars = ['time'] + global_vars
        parser, ref_data = self._init_random_parse(n_time, n_pts, n_ele, ele_size, field_vars, global_vars)
        self.assertEqual(len(goal_global_vars), len(parser.global_field_names))
        for gv in global_vars:
            self.assertIn(gv, parser.global_field_names)

    def test_get_correct_global_field_values(self):
        n_time = 10
        n_pts = 5
        n_ele = 1
        ele_size = 5
        field_vars = ['A', 'T']

        global_vars = ['total_Mac', 'total_and', 'total_cheese']
        goal_global_vars = ['time'] + global_vars
        parser, ref_data = self._init_random_parse(n_time, n_pts, n_ele, ele_size, field_vars, global_vars)
        global_values = parser.get_global_data()
        for gv_name in goal_global_vars:
            self.assert_close_arrays(ref_data[gv_name], global_values[gv_name])

    def test_get_correct_node_field_names(self):
        n_time = 10
        n_pts = 5
        n_ele = 1
        ele_size = 5
        field_vars = ['A', 'T']
        global_vars = ['total_heat']
        parser, ref_data = self._init_random_parse(n_time, n_pts, n_ele, ele_size, field_vars, global_vars)
        self.assertEqual(len(field_vars), len(parser.node_field_names))
        for fv in field_vars:
            self.assertIn(fv, parser.node_field_names)

        field_vars = ['A', 'T', 'G', 'as', 're']
        global_vars = ['total_heat']
        parser, ref_data = self._init_random_parse(n_time, n_pts, n_ele, ele_size, field_vars, global_vars)
        self.assertEqual(len(field_vars), len(parser.node_field_names))
        for fv in field_vars:
            self.assertIn(fv, parser.node_field_names)

    def test_get_correct_node_field_values(self):
        n_time = 10
        n_pts = 5
        n_ele = 1
        ele_size = 5
        field_vars = ['A', 'T']
        global_vars = ['total_heat']
        parser, ref_data = self._init_random_parse(n_time, n_pts, n_ele, ele_size, field_vars, global_vars)
        for i in range(n_time):
            parsed_data = parser.get_frame(i)
            for fv_name in field_vars:
                self.assert_close_arrays(parsed_data[fv_name], ref_data[fv_name][i,:])
            for pos_idx, pos_name in enumerate(['X', 'Y', 'Z']):
                self.assert_close_arrays(ref_data.skeleton.spatial_coords[:, pos_idx], parsed_data[pos_name])

    def test_get_correct_number_of_nodes(self):
        n_time = 10
        n_pts = 5
        n_ele = 1
        ele_size = 5
        field_vars = ['A', 'T']
        global_vars = ['total_heat']
        parser, ref_data = self._init_random_parse(n_time, n_pts, n_ele, ele_size, field_vars, global_vars)
        self.assertEqual(n_pts, parser.number_of_nodes)
        n_pts = 54
        parser, ref_data = self._init_random_parse(n_time, n_pts, n_ele, ele_size, field_vars, global_vars)
        self.assertEqual(n_pts, parser.number_of_nodes)

    def test_get_correct_number_of_elements(self):
        n_time = 10
        n_pts = 50
        n_ele = 11
        ele_size = 5
        field_vars = ['A', 'T']
        global_vars = ['total_heat']
        parser, ref_data = self._init_random_parse(n_time, n_pts, n_ele, ele_size, field_vars, global_vars)
        self.assertEqual(n_ele, parser.number_of_elements)
        n_ele = 23
        parser, ref_data = self._init_random_parse(n_time, n_pts, n_ele, ele_size, field_vars, global_vars)
        self.assertEqual(n_ele, parser.number_of_elements)