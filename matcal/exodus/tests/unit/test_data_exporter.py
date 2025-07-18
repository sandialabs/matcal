import numpy as np
import os

from matcal.core.constants import TIME_KEY
from matcal.core.tests.MatcalUnitTest import MatcalUnitTest

from matcal.full_field.data_exporter import MatCalFieldDataExporterIdentifier
from matcal.full_field.data_importer import FieldSeriesData
from matcal.full_field.tests.unit.test_mappers import FieldStatsExportTests

from matcal.exodus.data_exporter import exodus_field_data_exporter_function

from matcal.exodus.mesh_modifications import extract_exodus_mesh
from matcal.exodus.tests.unit.test_mesh_modifications import _create_field_data
from matcal.exodus.tests.utilities import test_support_files_dir

class TestMatCalFieldDataExporter(MatcalUnitTest):
    
    def setUp(self):
        super().setUp(__file__)

    def test_identify(self):
        exporter = MatCalFieldDataExporterIdentifier.identify("e")
        self.assertEqual(exodus_field_data_exporter_function, exporter)
        exporter = MatCalFieldDataExporterIdentifier.identify("g")
        self.assertEqual(exodus_field_data_exporter_function, exporter)
        exporter = MatCalFieldDataExporterIdentifier.identify("exo")
        self.assertEqual(exodus_field_data_exporter_function, exporter)

    def test_copy_mesh_and_store_node_and_global_data_user_specified_time_varname_with_time(self):
        mesh_name = os.path.join(test_support_files_dir, "test_mesh.g")
        mesh_skeleton = extract_exodus_mesh(mesh_name)
        time = np.linspace(0,10,20)
        field_data = _create_field_data(time, mesh_skeleton.spatial_coords, 
                                          "val", global_var=True)
        target_filename = "test_result.e"

        exodus_field_data_exporter_function(target_filename, field_data, 
                                            ["val", "val_gvar", "time"], mesh_name, 
                                            "val_gvar")
        
        mesh_data = FieldSeriesData(target_filename)
        self.assert_close_arrays(field_data["val_gvar"], mesh_data[TIME_KEY])
        for field in field_data.field_names:
            self.assertTrue(field in mesh_data.field_names)

    def test_copy_mesh_and_store_node_and_global_data_user_specified_time_varname_without_time(self):
        mesh_name = os.path.join(test_support_files_dir, "test_mesh.g")
        mesh_skeleton = extract_exodus_mesh(mesh_name)
        mesh_skeleton = extract_exodus_mesh(mesh_name)
        time = np.linspace(0,10,20)
        field_data = _create_field_data(time, mesh_skeleton.spatial_coords, 
                                          "val", global_var=True)
        target_filename = "test_result.e"
        field_data = field_data.remove_field(TIME_KEY)
        exodus_field_data_exporter_function(target_filename, field_data, 
                                            ["val", "val_gvar"], mesh_name, 
                                            "val_gvar")
        
        mesh_data = FieldSeriesData(target_filename)
        self.assert_close_arrays(field_data["val_gvar"], mesh_data[TIME_KEY])
        for field in field_data.field_names:
            self.assertTrue(field in mesh_data.field_names)
            
            
class TestFieldMapperExportE(FieldStatsExportTests.CommonExportTests):
    
    def setUp(self):
        super().setUp(__file__)
        
    @property
    def export_filename(self):
        return 'target.e'
        
    
class TestFieldMapperExportExo(FieldStatsExportTests.CommonExportTests):
    
    def setUp(self):
        super().setUp(__file__)
        
    @property
    def export_filename(self):
        return 'target.exo'

class TestFieldMapperExportG(FieldStatsExportTests.CommonExportTests):
    
    def setUp(self):
        super().setUp(__file__)
        
    @property
    def export_filename(self):
        return 'target.g'
        