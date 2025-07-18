import numpy as np
import os
import shutil

from matcal.core.constants import TIME_KEY
from matcal.core.data import DataCollection
from matcal.core.objective import ObjectiveCollection, ObjectiveSet
from matcal.core.tests.MatcalUnitTest import MatcalUnitTest

from matcal.exodus.library_importer import create_exodus_class_instance
from matcal.exodus.mesh_modifications import extract_exodus_mesh, store_information_on_mesh
from matcal.exodus.tests.utilities import test_support_files_dir

from matcal.full_field.data import convert_dictionary_to_field_data
from matcal.full_field.data_importer import FieldSeriesData
from matcal.full_field.objective import PolynomialHWDObjective, InterpolatedFullFieldObjective


def add_data_to_mesh_file(base_mesh_filename, target_mesh_filename, data_to_add, node_fieldnames):
    shutil.copy(base_mesh_filename, target_mesh_filename)
    exo_obj = create_exodus_class_instance(target_mesh_filename, mode='a', array_type='numpy')
    exo_obj = store_information_on_mesh(exo_obj, data_to_add, node_fieldnames)
    exo_obj.close()


def get_random_points(n_points, n_dim):
    p_array = np.random.uniform(-.5, .5, [n_points, n_dim])
    dim_names = ['x', 'y', 'z']
    points = {}
    for i in range(n_dim):
        points[dim_names[i]] = p_array[:,i]
    return points


class TestHWDObjectiveWithExodus(MatcalUnitTest):

    def setUp(self):
        super().setUp(__file__)
        path = self.get_current_files_path(__file__)
        self._gen_file = os.path.join(path, test_support_files_dir, "flat.g")
        self._gen_skeleton = extract_exodus_mesh(self._gen_file)

    def test_linear_with_surface_extraction(self):
        max_depth = 0
        max_poly = 2
        time_var = TIME_KEY
        dep_var = "temp"
        filename = "./linear.e"
        s_name = "FaCe"

        def linear_fun(x, y, t):
            out = np.zeros([len(t), len(x)])
            pos = x + 2 * y
            for t_idx, t_val  in enumerate(t):
                out[t_idx, :] = pos + t_val
            return out
        
        dc_exp, dc_sim = _make_data_collections(self._gen_file, self._gen_skeleton.spatial_coords, 
                                                time_var, dep_var, filename, linear_fun)

        hwdo = PolynomialHWDObjective(None, dep_var, max_depth=max_depth,  
                                      polynomial_order=max_poly)
        hwdo.extract_data_from_mesh_surface(s_name)
        residual = self._get_residual(dc_exp, dc_sim, hwdo)
        self.assert_close_arrays(np.zeros_like(residual), residual)

    def _get_residual(self, dc_exp, dc_sim, obj):
        obj_set = ObjectiveSet(ObjectiveCollection('test', obj), dc_exp, dc_exp.states)
        set_results = obj_set.calculate_objective_set_results(dc_sim)
        results = set_results[0][obj.name]
        residual = results.calibration_residuals
        return residual

    def test_linear_with_bad_surface_extraction_raise_key_error(self):
        max_depth = 0
        max_poly = 2
        time_var = TIME_KEY
        dep_var = "temp"
        filename = "./linear.e"
        s_name_wrong = "FaCeBAD"

        def linear_fun(x, y, t):
            out = np.zeros([len(t), len(x)])
            pos = x + 2 * y
            for t_idx, t_val  in enumerate(t):
                out[t_idx, :] = pos + t_val
            return out
        
        dc_exp, dc_sim = _make_data_collections(self._gen_file, self._gen_skeleton.spatial_coords, 
                                                time_var, dep_var, filename, linear_fun)

        hwdo = PolynomialHWDObjective(None, dep_var, max_depth=max_depth, polynomial_order=max_poly)
        hwdo.extract_data_from_mesh_surface(s_name_wrong)
        obj_set = ObjectiveSet(ObjectiveCollection('test', hwdo), dc_exp, dc_exp.states)
        with self.assertRaises(KeyError):
            set_results = obj_set.calculate_objective_set_results(dc_sim)

def _make_data_collections(gen_file, spatial_coords, time_var, dep_var, filename, linear_fun):
    ref_dict = get_random_points(20,2)
    ref_dict[time_var]= np.linspace(0, 10, 5)        
    ref_dict[dep_var] = linear_fun(ref_dict['x'], ref_dict['y'], ref_dict[time_var])
    ref = convert_dictionary_to_field_data(ref_dict, ['x','y'])
    ref.set_name("EXP")

    node_vars = [dep_var]
    test_dict = {time_var:np.linspace(0, 10, 8)}
    x = spatial_coords[:,0]
    y = spatial_coords[:,1]
    test_dict[dep_var] = linear_fun(x, y, test_dict[time_var])
    add_data_to_mesh_file(gen_file, filename, test_dict, node_vars)
    test = FieldSeriesData(filename)
    test.set_name("SIM")


    dc_exp = DataCollection('exp', ref)
    dc_sim = DataCollection('sim', test)
    return dc_exp,dc_sim
        
        
class TestInterpolatedFFWithExodus(MatcalUnitTest):

    def setUp(self):
        super().setUp(__file__)
        path = self.get_current_files_path(__file__)
        self._gen_file = os.path.join(path, test_support_files_dir, "flat.g")
        self._gen_skeleton = extract_exodus_mesh(self._gen_file)

    def _get_residual(self, dc_exp, dc_sim, obj):
        obj_set = ObjectiveSet(ObjectiveCollection('test', obj), dc_exp, dc_exp.states)
        set_results = obj_set.calculate_objective_set_results(dc_sim)
        results = set_results[obj.name]
        residual = results.calibration_residuals
        return residual

    def test_linear_with_surface_extraction(self):
        time_var = TIME_KEY
        dep_var = "temp"
        filename = "./linear.e"
        s_name = "FaCe"

        def linear_fun(x, y, t):
            out = np.zeros([len(t), len(x)])
            pos = x + 2 * y
            for t_idx, t_val  in enumerate(t):
                out[t_idx, :] = pos + t_val
            return out
        
        dc_exp, dc_sim = _make_data_collections(self._gen_file, self._gen_skeleton.spatial_coords,
                                                time_var, dep_var, filename, linear_fun)

        iffo = InterpolatedFullFieldObjective(self._gen_file, dep_var, fem_surface=s_name)
        obj_set = ObjectiveSet(ObjectiveCollection('test', iffo), dc_exp, dc_exp.states)
        set_results = obj_set.calculate_objective_set_results(dc_sim)
        results = set_results[0][iffo.name]
        residual = results.calibration_residuals
        self.assert_close_arrays(np.zeros_like(residual), residual)

