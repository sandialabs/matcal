from matcal.core.tests.MatcalUnitTest import MatcalUnitTest

import shutil

from matcal.sierra.models import *
from matcal.sierra.tests.platform_options import MatCalTestPlatformOptionsFunctionIdentifier
from matcal.sierra.tests.sierra_sm_models_for_tests import ( 
    UniaxialLoadingMaterialPointModelForTests)
from matcal.sierra.tests.utilities import (TEST_SUPPORT_FILES_FOLDER, 
    replace_string_in_file)

import matcal as mc


SET_PLATFORM_OPTIONS = MatCalTestPlatformOptionsFunctionIdentifier.identify()


class TestUserModels(MatcalUnitTest):

    def setUp(self):
        super().setUp(__file__)

    def test_aria_user_model_calibration(self):
        test_support_dir= os.path.join(TEST_SUPPORT_FILES_FOLDER,
                                     "user_defined_model_tests")
        goal_data = mc.FileData(os.path.join(test_support_dir, "user_defined_model_goal.csv"))
        input_file =os.path.join(test_support_dir, "conductivity.i")
        mesh_file = "square.g"
        mesh_file = os.path.join(test_support_dir, mesh_file)
        model = UserDefinedSierraModel('aria', input_file, mesh_file)
        model.continue_when_simulation_fails()
        SET_PLATFORM_OPTIONS(model)
        conductivity = mc.Parameter("K", 0, 5, 2.5)
        
        study = mc.GradientCalibrationStudy(conductivity)
        study.set_core_limit(32)
        objective= mc.CurveBasedInterpolatedObjective("time", "temperature")
        study.add_evaluation_set(model, objective, goal_data)

        results = study.launch()

        self.assertAlmostEqual(results.outcome["best:K"], 1, delta=1e6)

    def redundant_consider_removing_test_adagio_user_model_calibration(self):
        model_generator = UniaxialLoadingMaterialPointModelForTests()
        model = model_generator.init_model(plasticity=True)
        bc_data = model_generator.boundary_condition_data_sets[-1]
        model.add_boundary_condition_data(bc_data)
        state_name = bc_data.state_names[-1]
        state = bc_data.states[state_name]
        model.set_name("test_model")
        model.set_number_of_time_steps(50)
        pc = model_generator.get_material_parameter_collection()
        results_gold = model.run(state, pc)
        target_dir = model.get_target_dir_name(state)
        input_name = os.path.join(target_dir, "test_model.aprepro.i")
        replace_string_in_file(input_name, 
                               "yield stress = 250000000", 
                               "yield stress = {yield_stress}")
        mesh_name = "test_model.g"
        mesh_name_path = os.path.join(target_dir, mesh_name)
        shutil.copyfile(mesh_name_path, mesh_name)

        model = UserDefinedSierraModel('adagio', input_name, mesh_name)
        SET_PLATFORM_OPTIONS(model)

        yield_stress = mc.Parameter("yield_stress", 50e6, 600e6, 100e6)
        study = mc.GradientCalibrationStudy(yield_stress)
        objective = mc.CurveBasedInterpolatedObjective("true_strain", "true_stress")
        study.set_core_limit(32)
        study.add_evaluation_set(model, objective, results_gold.results_data)
        results = study.launch()
        goal_yield_stress = pc.get_current_value_dict()["yield_stress"]
        self.assertAlmostEqual(results.outcome["best:yield_stress"], 
                               goal_yield_stress, delta=goal_yield_stress*1e-6)

