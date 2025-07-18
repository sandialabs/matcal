from abc import abstractmethod
import os

from matcal import *

from matcal.core.tests.MatcalUnitTest import MatcalUnitTest

from matcal.full_field.TwoDimensionalFieldGrid import auto_generate_two_dimensional_field_grid
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
            mat_props = MatcalGeneratedModelForTestsBase.get_elastic_material_parameter_collection().get_current_value_dict()
            mat_props.update(MatcalGeneratedModelForTestsBase.get_material_properties())
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

        def prepare_plastic_gold_results(self):
            self.set_rectangle_filenames_and_paths()
            gold_mesh_str = read_file_lines(self.solid_aprepro)+ \
                read_file_lines(self.journal_filename)
            self.constants["mat_model"] = "j2_plasticity"

            create_goal_user_model_simulation_results(self.input_filename, self.solid_mesh_filename, 
                                        self.gold_plastic_results_filename, self.solid_aprepro,
                                        self.material_file, 
                                        mesh_str=gold_mesh_str,
                                        run_dir=self.goal_files_dir, 
                                        constants=self.constants, cores=4,
                                        **self.goal_param_vals)
            
            field_data = FieldSeriesData(self.gold_plastic_results_filename)
            field_data.rename_field("displacement_x", "U")
            field_data.rename_field("displacement_y", "V")
            return field_data

        def verification_test_basic_plastic_vfm_move_to_docs(self):
            field_data = self.prepare_plastic_gold_results()

            number_of_nodes_x = 5
            number_of_nodes_y = 10
            thickness = .001
            mesh_skeleton = auto_generate_two_dimensional_field_grid(number_of_nodes_x, number_of_nodes_y, field_data)

            mat = Material("matcal_test", self.material_file, "j2_plasticity")

            vfm_model = self._VFM_model_type(mat, mesh_skeleton, thickness)
            mat_props = MatcalGeneratedModelForTestsBase.get_material_parameter_collection().get_current_value_dict()
            mat_props.update(MatcalGeneratedModelForTestsBase.get_material_properties())
            vfm_model.add_constants(**mat_props, coupling="uncoupled")
            vfm_model.set_number_of_cores(1)
            vfm_model.add_boundary_condition_data(field_data)
            vfm_model.set_displacement_field_names("U", "V")
            vfm_model.use_under_integrated_element()

            vfm_objective = MechanicalVFMObjective()

            yield_stress = Parameter("yield_stress", 100e6, 500e6, 200e6)
            A = Parameter("A", 1000e6, 5000e6, 4000e6)
            b = Parameter("b", 0, 10, 5+0.001* np.random.uniform(0,1))

            calibration = GradientCalibrationStudy(yield_stress, A, b)
            calibration.add_evaluation_set(vfm_model, vfm_objective, field_data)
            calibration.set_core_limit(32)
            calibration.set_convergence_tolerance(1e-12)

            results = calibration.launch()

            goal_yield_stress = mat_props["yield_stress"]
            goal_A = mat_props["A"]
            goal_b = mat_props["b"]
            
            self.assertAlmostEqual(results.outcome["best:yield_stress"], goal_yield_stress, delta=goal_yield_stress*self.relative_error_tol)
            self.assertAlmostEqual(results.outcome["best:A"], goal_A, delta=goal_A*self.relative_error_tol)
            self.assertAlmostEqual(results.outcome["best:b"], goal_b, delta=goal_A*self.relative_error_tol)

        def set_complex_filenames_and_paths(self):
            self.test_files_subdir =  "complex_vfm_test_files"
            self.template_files_dir = os.path.join(TEST_SUPPORT_FILES_FOLDER, self.test_files_subdir)
            self.input_filename = os.path.join(self.template_files_dir, "complex_vfm_gold.i")
            self.solid_aprepro = os.path.join(self.template_files_dir, "solid_mesh.inc")

            self.shell_aprepro = os.path.join(self.template_files_dir, "shell_mesh.inc")
            self.journal_filename = os.path.join(self.template_files_dir, "complex_vfm_mesh.jou")
            self.constants["aprepro_file"] = self.shell_aprepro

            self.goal_files_dir = os.path.join(GENERATED_TEST_DATA_FOLDER, self.test_files_subdir)
            self.results_filename = "results.e"
            self.gold_results_filename = os.path.join(self.goal_files_dir, self.results_filename)
            
            self.shell_mesh_filename = os.path.join(self.goal_files_dir, "complex_vfm_mesh_shell.g")
            self.solid_mesh_filename = os.path.join(self.goal_files_dir, "complex_vfm_mesh.g")

        def prepare_complex_gold_results(self):
            self.set_complex_filenames_and_paths()
            gold_mesh_str = read_file_lines(self.shell_aprepro)+ \
                read_file_lines(self.journal_filename)
            self.constants["mat_model"] = "j2_plasticity"

            create_goal_user_model_simulation_results(self.input_filename, self.shell_mesh_filename, 
                                        self.gold_results_filename, self.shell_aprepro,
                                        self.material_file, 
                                        mesh_str=gold_mesh_str,
                                        run_dir=self.goal_files_dir, 
                                        constants=self.constants, cores=8,
                                        **self.goal_param_vals)
            
            field_data = FieldSeriesData(self.gold_results_filename)
            field_data = field_data[field_data["time"] <= 8.5]
            return field_data

        def verification_test_complex_shape_plastic_vfm_do_not_run_every_time_make_example(self):
            self.relative_error_tol = 1e-2
            field_data = self.prepare_complex_gold_results()
            
            mat = Material("matcal_test", self.material_file, "j2_plasticity")

            vfm_model = self._VFM_model_type(mat, self.shell_mesh_filename, thickness=0.0625*0.0254)
            mat_props = MatcalGeneratedModelForTestsBase.get_material_parameter_collection().get_current_value_dict()
            mat_props.update(MatcalGeneratedModelForTestsBase.get_material_properties())
            vfm_model.add_constants(**mat_props)
            vfm_model.set_number_of_cores(36)
            vfm_model.add_boundary_condition_data(field_data)
            vfm_model.set_displacement_field_names("U", "V")
            vfm_model.set_mapping_parameters(2,1.1) ### need weird mapping parameters because it 
            # has coarse data that it is calibrating against. The experimental data mesh is 
            # similar to the model being calibrated to it. 
            vfm_model.set_number_of_time_steps(400)

            vfm_objective = MechanicalVFMObjective()

            yield_stress = Parameter("yield_stress", 100e6, 500e6, 250e6*1.025)
            A = Parameter("A", 1000e6, 5000e6, 2500e6*1.025)#4000e6)
            b = Parameter("b", 0, 10, 2*1.02+0.005* np.random.uniform(0,1))#5

            calibration = GradientCalibrationStudy(yield_stress,  A, b)
            calibration.add_evaluation_set(vfm_model, vfm_objective, field_data)
            calibration.set_core_limit(112)
            calibration.set_step_size(1e-6)
            calibration.set_convergence_tolerance(1e-12)

            results = calibration.launch()

            goal_yield_stress = mat_props["yield_stress"]
            goal_A = mat_props["A"]
            goal_b = mat_props["b"]

            self.assertAlmostEqual(results.outcome["best:yield_stress"], goal_yield_stress, delta=goal_yield_stress*self.relative_error_tol)
            self.assertAlmostEqual(results.outcome["best:A"], goal_A, delta=goal_A*self.relative_error_tol)
            self.assertAlmostEqual(results.outcome["best:b"], goal_b, delta=goal_b*self.relative_error_tol)

class HexVFMModelTests(VFMCalibrationTestsBase.CommonTests):
    _VFM_model_type = VFMUniaxialTensionHexModel

class ConnectedHexVFMModelTests(VFMCalibrationTestsBase.CommonTests):
    _VFM_model_type = VFMUniaxialTensionConnectedHexModel

    def test_basic_plastic_vfm(self):
        """Not tested to reduce computation time for production tests."""