import numpy as np
import os
import shutil

from matcal.core.tests.MatcalUnitTest import MatcalUnitTest
from matcal.exodus.mesh_modifications import _ExodusFieldInterpPreprocessor
from matcal.exodus.tests.utilities import test_support_files_dir, _open_mesh
from matcal.full_field.data_importer import FieldSeriesData, ImportedTwoDimensionalMesh



class TestImportedFieldSeriesWithIntepolationComplexGeometry(MatcalUnitTest):

    def setUp(self):
        super().setUp(__file__)
        self._fine_test_mesh_name = "fine_complex_test.g"
        self._coarse_test_mesh_name = "coarse_complex_test.g"

        self._fine_test_mesh = os.path.join(test_support_files_dir, 
                                            self._fine_test_mesh_name )
        self._coarse_test_mesh = os.path.join(test_support_files_dir, 
                                              self._coarse_test_mesh_name )
        
        self._source_mesh = self._prepare_mesh_series_data(self._fine_test_mesh)
        self._source_mesh.close()
        self._series_data = FieldSeriesData(self._fine_test_mesh, file_type="e")

    def _prepare_mesh_series_data(self, filename):
        mesh = _open_mesh(filename, open_mode="a")
        node_variable_names = ["x", "y"]
        mesh = self._initialize_node_variables(mesh, node_variable_names)
        times = np.linspace(0, 20, 10)
        for time_step, time in enumerate(times):
            time_step += 1
            mesh.put_time(time_step, time)
            for node_variable_name in node_variable_names:
                x_coords, y_coords, z_coords = mesh.get_coords()
                new_mesh_node_variable_values = node_var_funcs[node_variable_name](x_coords, y_coords, time)
                mesh.put_node_variable_values(node_variable_name, time_step, new_mesh_node_variable_values)
        return mesh
    
    def _initialize_node_variables(self, mesh_object, node_variable_names):
        mesh_object.set_node_variable_number(len(node_variable_names))
        for node_variable_index, node_variable_name in enumerate(node_variable_names):
                mesh_object.put_node_variable_name(node_variable_name, node_variable_index+1)
        return mesh_object
 
    def test_interpolate_fields_same_mesh_egc(self):
        from matcal.exodus.geometry import ExodusHexGeometryCreator
        target = ImportedTwoDimensionalMesh(self._fine_test_mesh)
        hex_mesh_filename = "hex_"+self._fine_test_mesh_name
        geo_params = {"reference_mesh_grid":target, "thickness":0.1}
        egc = ExodusHexGeometryCreator(hex_mesh_filename, geo_params)
        egc.create_mesh()
        shutil.copy(hex_mesh_filename, "err_"+hex_mesh_filename)

        efpp = _ExodusFieldInterpPreprocessor()
        efpp.process("./", hex_mesh_filename, self._series_data,
                      ["x", "y"], 2, 2)

        efpp_result_exo = _open_mesh(hex_mesh_filename, "r")

        x, y, z = efpp_result_exo.get_coords()
        times = efpp_result_exo.get_times()

        for time_step, time in enumerate(times):
            time_step += 1
            for field in ["x", "y"]:
                results = efpp_result_exo.get_node_variable_values(field, time_step)
                self.assert_close_arrays(results, node_var_funcs[field](x, y, time))
        efpp_result_exo.close()

    def test_interpolate_fields_coarse_mesh_egc(self):
        target = ImportedTwoDimensionalMesh(self._coarse_test_mesh)
        from matcal.exodus.geometry import ExodusHexGeometryCreator
        hex_mesh_filename = "hex_"+self._coarse_test_mesh_name
        geo_params = {"reference_mesh_grid":target, "thickness":0.1}
        egc = ExodusHexGeometryCreator(hex_mesh_filename, geo_params)
        egc.create_mesh()
        shutil.copy(hex_mesh_filename, "err_"+hex_mesh_filename)

        efpp = _ExodusFieldInterpPreprocessor()
        efpp.process("./", hex_mesh_filename, self._series_data,
                      ["x", "y"], 2, 2)

        efpp_result_exo = _open_mesh(hex_mesh_filename, "r")

        x, y, z = efpp_result_exo.get_coords()
        times = efpp_result_exo.get_times()

        for time_step, time in enumerate(times):
            time_step += 1
            for field in ["x", "y"]:
                results = efpp_result_exo.get_node_variable_values(field, time_step)
                self.assert_close_arrays(results, node_var_funcs[field](x, y, time))
        efpp_result_exo.close()

def x_function(x_coords, y_coords, time):
    return (x_coords*y_coords+3*y_coords+1e-10)*(time/20+1)

def y_function(x_coords, y_coords, time):
    return (x_coords*x_coords+x_coords*y_coords+2*y_coords+1e-10)*(time/20+1)

node_var_funcs = {"x": x_function, 
                  "y": y_function} 