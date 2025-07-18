import numpy as np
import os

from matcal.core.serializer_wrapper import json_serializer
from matcal.core.tests.MatcalUnitTest import MatcalUnitTest

from matcal.full_field.data import convert_dictionary_to_field_data
from matcal.full_field.data_exporter import (
                                             export_full_field_data_to_json, 
                                             serialize_full_field_data, 
                                             MatCalFieldDataExporterIdentifier)


class TestJSONFieldDataExporter(MatcalUnitTest):

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

    def test_export_ff_data_file_exits(self):
        ff_data = self._make_simple_mesh_with_info()
        export_name = "results.json"
        export_full_field_data_to_json(export_name, ff_data)
        self.assertTrue(os.path.exists(export_name))

    def test_export_ff_data_and_load_json_fields(self):
        ff_data = self._make_simple_mesh_with_info()
        export_name = 'results.json'
        export_full_field_data_to_json(export_name, ff_data)
        with open(export_name, 'r') as f:
            read_data = json_serializer.load(f)
        goal_fields = ['T', 'time', 'spatial_coords', 'connectivity', 'subset_name', 'surfaces']
        for field in goal_fields:
            self.assertIn(field, read_data.keys())

    def test_export_ff_data_confirm_values(self):
        ff_data = self._make_simple_mesh_with_info()
        export_name = 'results.json'
        export_full_field_data_to_json(export_name, ff_data)
        with open(export_name, 'r') as f:
            read_data = json_serializer.load(f)
        goal = {'T':ff_data['T'], 'time':ff_data['time'], 'spatial_coords':ff_data.skeleton.spatial_coords,
                 'connectivity':ff_data.skeleton.connectivity}
        subset_name = read_data.pop('subset_name')
        self.assertIsNone(subset_name)
        surfaces = read_data.pop('surfaces')
        empty_surfaces = {}
        self.assertEqual(surfaces, empty_surfaces)
        
        self.assert_close_dicts_or_data(goal, read_data, show_arrays=False)

    def test_serialize_ff_data(self):
        ff_data = self._make_simple_mesh_with_info()
        serial_data = serialize_full_field_data(ff_data)
        goal_fields = ['T', 'time', 'spatial_coords', 'connectivity', 'subset_name', 'surfaces']
        for field in goal_fields:
            self.assertIn(field, serial_data.keys())
    
class TestMatCalFieldDataExporter(MatcalUnitTest):
    
    def setUp(self):
        super().setUp(__file__)
 
    def test_identify(self):
        exporter = MatCalFieldDataExporterIdentifier.identify("json")
        self.assertEqual(export_full_field_data_to_json, exporter)