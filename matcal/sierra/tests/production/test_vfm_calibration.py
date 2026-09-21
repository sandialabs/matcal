from abc import abstractmethod
import os
import numpy as np

from matcal import *

from matcal.core.tests.MatcalUnitTest import MatcalUnitTest

from matcal.full_field.objective import MechanicalVFMObjective

from matcal.sierra.models import *
from matcal.sierra.tests.sierra_sm_models_for_tests import MatcalGeneratedModelForTestsBase
from matcal.sierra.tests.utilities import (TEST_SUPPORT_FILES_FOLDER, GENERATED_TEST_DATA_FOLDER, 
    create_goal_user_model_simulation_results, make_mesh_from_string_or_journal, 
        write_j2_plasticity_material_file, read_file_lines)


class VFMCalibrationTestsBase():
    def __init__():
        pass
    class CommonTests(MatcalUnitTest):

        @property
        @abstractmethod
        def _VFM_model_type(self):
            """"""

        def setUp(self):
            super().setUp(__file__)
            self.relative_error_tol = 1e-6
            self.material_file = write_j2_plasticity_material_file()
            self.constants = MatcalGeneratedModelForTestsBase.get_material_properties()
            self.constants["coupling"] = "uncoupled"
            self.constants["mat_model"] = "linear_elastic"
            self.goal_param_vals =  MatcalGeneratedModelForTestsBase.get_material_parameter_collection().get_current_value_dict()

        def set_rectangle_filenames_and_paths(self):
            self.test_files_subdir =  "rectangle_vfm_test_files"
            self.template_files_dir = os.path.join(TEST_SUPPORT_FILES_FOLDER, self.test_files_subdir)
            self.input_filename = os.path.join(self.template_files_dir, "rectangle_input.i")
            self.solid_aprepro = os.path.join(self.template_files_dir, "make_fine_thinner_solid.inc")
            self.constants["aprepro_file"] = self.solid_aprepro

            self.shell_aprepro = os.path.join(self.template_files_dir, "make_coarse_wider_surface.inc")
            self.journal_filename = os.path.join(self.template_files_dir, "make_rect_mesh_needs_N_and_solid_mesh.jou")

            self.goal_files_dir = os.path.join(GENERATED_TEST_DATA_FOLDER, self.test_files_subdir)
            self.elastic_results_filename = "elastic_results.e"
            self.plastic_results_filename = "plastic_results.e"
            self.gold_elastic_results_filename = os.path.join(self.goal_files_dir, self.elastic_results_filename)
            self.gold_plastic_results_filename = os.path.join(self.goal_files_dir, self.plastic_results_filename)
            
            self.shell_mesh_filename = os.path.join(self.goal_files_dir, "thin_rect_surface.g")
            self.solid_mesh_filename = os.path.join(self.goal_files_dir, "thin_rect.g")

        def prepare_elastic_gold_results(self):
            self.set_rectangle_filenames_and_paths()
            gold_mesh_str = read_file_lines(self.solid_aprepro)+ \
                read_file_lines(self.journal_filename)
            create_goal_user_model_simulation_results(self.input_filename, self.solid_mesh_filename, 
                                        self.gold_elastic_results_filename, self.solid_aprepro,
                                        self.material_file, 
                                        mesh_str=gold_mesh_str,
                                        run_dir=self.goal_files_dir, 
                                        constants=self.constants, cores=2,
                                        **self.goal_param_vals)
            
            field_data = FieldSeriesData(self.gold_elastic_results_filename)
            field_data.rename_field("displacement_x", "U")
            field_data.rename_field("displacement_y", "V")
            return field_data

        def test_elastic_with_exterior_dic_data(self):
            field_data = self.prepare_elastic_gold_results()

            thickness = .001
            scaled_width = 1.05 #made mesh 1.05X wider, so the load should be scale similarly
            field_data['load'] = scaled_width * field_data['load']

            mat = Material("matcal_test", self.material_file, self.constants["mat_model"])
            
            shell_mesh_str = read_file_lines(self.shell_aprepro) + \
                read_file_lines(self.journal_filename)
            make_mesh_from_string_or_journal(self.shell_mesh_filename, mesh_str=shell_mesh_str)

            vfm_model = self._VFM_model_type(mat, self.shell_mesh_filename, thickness)
            vfm_model.add_boundary_condition_data(field_data)
            plasticity_props = MatcalGeneratedModelForTestsBase.get_material_parameter_collection().get_current_value_dict()
            mat_props = MatcalGeneratedModelForTestsBase.get_elastic_material_parameter_collection().get_current_value_dict()
            mat_props.update(MatcalGeneratedModelForTestsBase.get_material_properties())
            mat_props.update(plasticity_props)
            goal_E = mat_props.pop('elastic_modulus')
            goal_nu = mat_props.pop('nu')
            vfm_model.add_constants(**mat_props, coupling="uncoupled")
            vfm_model.set_number_of_cores(1)
            vfm_model.set_displacement_field_names("U", "V")
            vfm_model.use_under_integrated_element()

            vfm_objective = MechanicalVFMObjective()

            e_mod = Parameter("elastic_modulus", 1e9, 500e9, 100e9)
            nu = Parameter("nu", 0.1, 0.4, 0.2+0.001* np.random.uniform(0,1))

            calibration = GradientCalibrationStudy(e_mod, nu)
            calibration.add_evaluation_set(vfm_model, vfm_objective, field_data)
            calibration.set_core_limit(32)
            calibration.set_convergence_tolerance(1e-12)

            results = calibration.launch()


            self.assertAlmostEqual(results.outcome["best:elastic_modulus"], goal_E, delta=goal_E*self.relative_error_tol)
            self.assertAlmostEqual(results.outcome["best:nu"], goal_nu, delta=goal_nu*self.relative_error_tol)

class HexVFMModelTests(VFMCalibrationTestsBase.CommonTests):
    _VFM_model_type = VFMUniaxialTensionHexModel

class ConnectedHexVFMModelTests(VFMCalibrationTestsBase.CommonTests):
    _VFM_model_type = VFMUniaxialTensionConnectedHexModel

    def test_basic_plastic_vfm(self):
        """Not tested to reduce computation time for production tests."""