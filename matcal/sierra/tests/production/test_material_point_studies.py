## want this to kind of look like an input a user would write
import os
from copy import deepcopy

from matcal.core.constants import (MATCAL_TEMPLATE_DIRECTORY, IN_PROGRESS_RESULTS_FILENAME, 
                                   EVALUATION_EXTENSION)
from matcal.core.calibration_studies import ScipyLeastSquaresStudy, ScipyMinimizeStudy

from matcal.core.logger import matcal_print_message
from matcal.core.tests.MatcalUnitTest import MatcalUnitTest

from matcal.sierra.models import *
from matcal.sierra.tests.sierra_sm_models_for_tests import (
    UniaxialLoadingMaterialPointModelForTests, 
    MatcalGeneratedModelForTestsBase)
from matcal.sierra.tests.utilities import (create_goal_model_simulation_results, 
    GENERATED_TEST_DATA_FOLDER)

from matcal import *
import time


def param_preprocessor_func(params):
    params["elastic_modulus"] = params["elastic_modulus"]*2
    params["test_func_values"] = """0.0, 0.0
            1.0,2.0
            2.0, 4.0"""
    return params

def my_function(**param_dict):
    import numpy as np
    if "restart_kill" in param_dict:
        matcal_print_message(param_dict)
        if param_dict['elastic_modulus'] < 2.05e+11:
            matcal_print_message("Emod close to sol:", param_dict['elastic_modulus'] < 2.05e+11)
            matcal_print_message("Emod:",param_dict['elastic_modulus'])
            
            return None
    return {"true_strain": np.log(1.5), "true_stress": 0}

MODEL_GENERATOR = UniaxialLoadingMaterialPointModelForTests()
BC_DATA = MODEL_GENERATOR.boundary_condition_data_sets
RESULTS_SUB_DIR = os.path.join(GENERATED_TEST_DATA_FOLDER, "mat_point_study_test_files")
if not os.path.exists(RESULTS_SUB_DIR):
    os.mkdir(RESULTS_SUB_DIR)

PC  = MatcalGeneratedModelForTestsBase.get_material_parameter_collection()
MODEL_CONSTANTS = MatcalGeneratedModelForTestsBase.get_material_properties()
EPC = MatcalGeneratedModelForTestsBase.get_elastic_material_parameter_collection()

def get_uniaxial_material_point_model(bc_data=None, plasticity=False):
    if bc_data is None:
        bc_data = MODEL_GENERATOR.boundary_condition_data_sets
    model = MODEL_GENERATOR.init_model(plasticity)
    model.add_boundary_condition_data(bc_data)
    return model

def create_simple_mat_point_model_and_results_1():
    data_dc = BC_DATA[-1]
    state_name = data_dc.state_names[-1]
    state = data_dc.states[state_name]
    data = data_dc[state_name][0]
    model = get_uniaxial_material_point_model(data)
    target_dir = os.path.join(RESULTS_SUB_DIR, f"simple_cal_state_{state.name}")
    target_results = os.path.join(target_dir, "results.csv")
    create_goal_model_simulation_results(model, target_results, run_dir=target_dir,
                                            state=state, **MODEL_CONSTANTS)
    goal_data = FileData(target_results, state=state)

    return goal_data, model

def create_simple_mat_point_model_and_results_2():
    data_dc = BC_DATA[-1]
    state_name = data_dc.state_names[-1]
    state = data_dc.states[state_name]
    data = data_dc[state_name][0]
    model = get_uniaxial_material_point_model(data)

    target_dir = os.path.join(RESULTS_SUB_DIR, f"simple_cal_state_{state.name}_2")
    target_results = os.path.join(target_dir, "results.csv")
    constants = deepcopy(MODEL_CONSTANTS)
    constants["elastic_modulus"] *= 1.2
    create_goal_model_simulation_results(model, target_results, run_dir=target_dir,
                                            state=state, **constants)
    goal_data = FileData(target_results, state=state)

    return goal_data, model, constants

def create_simple_mat_point_model_and_results_3():
    data_dc = BC_DATA[-1]
    state_name = data_dc.state_names[-1]
    state = data_dc.states[state_name]
    data = data_dc[state_name][0]
    model = get_uniaxial_material_point_model(data)

    target_dir = os.path.join(RESULTS_SUB_DIR, f"simple_cal_state_{state.name}_3")
    target_results = os.path.join(target_dir, "results.csv")
    constants = deepcopy(MODEL_CONSTANTS)
    constants["elastic_modulus"] *= 1.5
    create_goal_model_simulation_results(model, target_results, run_dir=target_dir,
                                            state=state, **constants)
    goal_data = FileData(target_results, state=state)

    return goal_data, model, constants

def create_plastic_simple_mat_point_model_and_results_two_state():
    data_dc = BC_DATA[-1]
    model = get_uniaxial_material_point_model(data_dc, plasticity=True)

    goal_dc = DataCollection("test goal")
    for state in data_dc.keys():
        target_dir = os.path.join(RESULTS_SUB_DIR, f"plastic_simple_cal_state_{state.name}")
        target_results = os.path.join(target_dir, "results.csv")
        create_goal_model_simulation_results(model, target_results, run_dir=target_dir,
                                             state=state, **MODEL_CONSTANTS, **PC.get_current_value_dict())
        goal_dc.add(FileData(target_results, state=state))
    return goal_dc, model

class MaterialPointModelStudyTests(MatcalUnitTest):

    def setUp(self):
        super().setUp(__file__)
        self.error_tol = 1e-6

    def test_gradient_calibration_simple_set_working_dir(self):
        goal_data, model = create_simple_mat_point_model_and_results_1()

        objective = CurveBasedInterpolatedObjective("true_strain", "true_stress")
        emod_cur_val = EPC.get_current_value_dict()["elastic_modulus"]
        EPC.update_parameters(elastic_modulus=emod_cur_val*(1+1e-2*np.random.uniform()))
        calibration = GradientCalibrationStudy(EPC["elastic_modulus"])
        calibration.add_evaluation_set(model, objective, goal_data)
        study_dir = "test_work_dir"
        calibration.set_working_directory(study_dir)
        calibration.set_core_limit(32)
        results = calibration.launch()
        goal_elastic_mod = MODEL_CONSTANTS["elastic_modulus"]
        self.assertAlmostEqual(results.outcome["best:elastic_modulus"], 
                               goal_elastic_mod, 
                               delta=goal_elastic_mod * self.error_tol)
        self.assertEqual(os.getcwd(), calibration._initial_directory)
        self.assertNotEqual(os.getcwd(), calibration._working_directory)
        self.assertTrue(os.path.exists(study_dir))
        self.assertTrue(os.path.exists(os.path.join(study_dir, "dakota.out")))
        self.assertTrue(os.path.exists(os.path.join(study_dir, MATCAL_TEMPLATE_DIRECTORY)))
        self.assertTrue(os.path.exists(calibration.final_results_filename))


    def test_LHS_sampling_with_restart(self):
        goal_data, model = create_simple_mat_point_model_and_results_1()

        objective = SimulationResultsSynchronizer("true_strain", 
                                        goal_data['true_strain'], "true_stress")
        emod_cur_val = EPC.get_current_value_dict()["elastic_modulus"]
        new_value = emod_cur_val*(1+1e-2*np.random.uniform())
        EPC.update_parameters(elastic_modulus=new_value)
        
        n_cores = 8
        seed = 513561
        n_samples = 8

        lhs = LhsSensitivityStudy(EPC['elastic_modulus'])
        lhs.add_evaluation_set(model, objective, goal_data)
        lhs.set_core_limit(8)
        lhs.set_random_seed(seed)
        lhs.set_number_of_samples(n_samples)
        lhs._for_testing_fail_after_first_batch()

        base_time_start = time.time()
        try:
            base_results = lhs.launch()
        except Exception:
            pass
        base_time_delta = time.time() - base_time_start
        
        # remove the dakota restart so it just uses the matcal restart
        import os
        os.system('rm -f dakota.* *.joblib')
        #os.remove('dakota.rst')
        import gc
        gc.collect()

        r_lhs = LhsSensitivityStudy(EPC['elastic_modulus'])
        r_lhs.add_evaluation_set(model, objective, goal_data)
        r_lhs.set_core_limit(8)
        r_lhs.set_random_seed(seed)
        r_lhs.set_number_of_samples(n_samples)
        r_lhs.restart(None)

        restart_time_start = time.time()
        restart_results = r_lhs.launch()
        restart_time_delta = time.time() - restart_time_start

        time_rato = base_time_delta / restart_time_delta
        min_time_ratio = 10
        self.assertGreaterEqual(time_rato, min_time_ratio)

    def redundant_maybe_mv_to_docs_test_gradient_calibration_simple_set_working_dir_scipy_least_squares(self):
        goal_data, model = create_simple_mat_point_model_and_results_1()

        objective = CurveBasedInterpolatedObjective("true_strain", "true_stress")
        emod_cur_val = EPC.get_current_value_dict()["elastic_modulus"]
        EPC.update_parameters(elastic_modulus=emod_cur_val*(1+1e-2*np.random.uniform()))
        calibration = ScipyLeastSquaresStudy(EPC["elastic_modulus"])
        calibration.add_evaluation_set(model, objective, goal_data)
        study_dir = "test_work_dir"
        calibration.set_working_directory(study_dir)
        calibration.set_core_limit(32)
        results = calibration.launch()
        goal_elastic_mod = MODEL_CONSTANTS["elastic_modulus"]
        self.assertAlmostEqual(results.best.elastic_modulus, 
                               goal_elastic_mod, 
                               delta=goal_elastic_mod * self.error_tol)
        self.assertEqual(os.getcwd(), calibration._initial_directory)
        self.assertNotEqual(os.getcwd(), calibration._working_directory)
        self.assertTrue(os.path.exists(study_dir))
        self.assertTrue(os.path.exists(os.path.join(study_dir, MATCAL_TEMPLATE_DIRECTORY)))
        self.assertTrue(os.path.exists(calibration.final_results_filename))
    
    def test_gradient_calibration_simple_set_working_dir_scipy_minimize(self):
        goal_data, model = create_simple_mat_point_model_and_results_1()
        self.error_tol = 1e-4

        objective = CurveBasedInterpolatedObjective("true_strain", "true_stress")
        emod_cur_val = EPC.get_current_value_dict()["elastic_modulus"]
        EPC.update_parameters(elastic_modulus=emod_cur_val*(1+1e-2*np.random.uniform()))
        cal_param = Parameter("elastic_modulus", 100, 400)
        calibration = ScipyMinimizeStudy(cal_param)
        def param_preproc(params):
            params['elastic_modulus'] *=1e9
            return params
        
        param_preproc = UserDefinedParameterPreprocessor(param_preproc)
        calibration.add_evaluation_set(model, objective, goal_data)
        calibration.add_parameter_preprocessor(param_preproc)
        study_dir = "test_work_dir"
        calibration.set_working_directory(study_dir)
        calibration.set_core_limit(32)
        results = calibration.launch()
        goal_elastic_mod = MODEL_CONSTANTS["elastic_modulus"]
        self.assertAlmostEqual(results.best.elastic_modulus*1e9, 
                               goal_elastic_mod, 
                               delta=goal_elastic_mod * self.error_tol)
        self.assertEqual(os.getcwd(), calibration._initial_directory)
        self.assertNotEqual(os.getcwd(), calibration._working_directory)
        self.assertTrue(os.path.exists(study_dir))
        self.assertTrue(os.path.exists(os.path.join(study_dir, MATCAL_TEMPLATE_DIRECTORY)))
        self.assertTrue(os.path.exists(calibration.final_results_filename))

    def test_gradient_calibration_simple_with_param_preprocessor(self):
        param_preprocessor = UserDefinedParameterPreprocessor(param_preprocessor_func)

        goal_data, model = create_simple_mat_point_model_and_results_1()
        objective = CurveBasedInterpolatedObjective("true_strain", "true_stress")
        emod_cur_val = EPC.get_current_value_dict()["elastic_modulus"]
        EPC.update_parameters(elastic_modulus=emod_cur_val*(1+1e-2*np.random.uniform()))
        calibration = GradientCalibrationStudy(EPC["elastic_modulus"])
        calibration.add_evaluation_set(model, objective, goal_data)
        calibration.add_parameter_preprocessor(param_preprocessor)
        calibration.set_core_limit(32)
        results = calibration.launch()
        goal_elastic_mod = MODEL_CONSTANTS["elastic_modulus"]/2
        self.assertAlmostEqual(results.best.elastic_modulus, 
                               goal_elastic_mod, 
                               delta=goal_elastic_mod * self.error_tol)

    def test_two_states_gradient_calibration_simple(self):
        goal_dc, model = create_plastic_simple_mat_point_model_and_results_two_state()
        model.add_constants(**MODEL_CONSTANTS)
        model.add_constants(**PC.get_current_value_dict())
        yield_stress = Parameter("yield_stress", 100e6, 600e6, current_value=200e6)
        calibration = GradientCalibrationStudy(yield_stress)
        objective = CurveBasedInterpolatedObjective('true_strain', 'true_stress')
        calibration.add_evaluation_set(model, objective, goal_dc)
        calibration.set_core_limit(32)
        results = calibration.launch()
        goal_result = PC.get_current_value_dict()["yield_stress"]
        self.assertAlmostEqual(results.best.yield_stress, 
                               goal_result, delta=goal_result * self.error_tol)

    def test_gradient_calibration_simple_two_parameters_three_data(self):
        goal_data, model = create_simple_mat_point_model_and_results_1()
        goal_data2, model, constants = create_simple_mat_point_model_and_results_2()
        goal_data3, model, constants2 = create_simple_mat_point_model_and_results_3()

        objective = CurveBasedInterpolatedObjective("true_strain", "true_stress", "contraction")
        calibration = GradientCalibrationStudy(EPC)
        dc = DataCollection("test", goal_data, goal_data2, goal_data3)

        calibration.add_evaluation_set(model, objective, dc)
        calibration.set_core_limit(32)
        results = calibration.launch()
        goal_elastic_mod = (MODEL_CONSTANTS["elastic_modulus"]+
                            constants["elastic_modulus"]+
                            constants2["elastic_modulus"])/3
        
        self.assertAlmostEqual(results.outcome["best:elastic_modulus"], 
                               goal_elastic_mod, 
                               delta=goal_elastic_mod * self.error_tol)
        self.assertAlmostEqual(results.outcome["best:nu"], MODEL_CONSTANTS["nu"])

    def test_gradient_calibration_simple_set_working_dir_with_restart(self):
        goal_data, model = create_simple_mat_point_model_and_results_1()
        curve_objective = CurveBasedInterpolatedObjective("true_strain", "true_stress")

        python_model = PythonModel(my_function)
        python_model.add_constants(restart_kill="True")
        objective = Objective("true_strain")
        objective.set_qoi_extractors(MaxExtractor("true_strain"))

        emod_cur_val = EPC.get_current_value_dict()["elastic_modulus"]

        restart_filename = "my_restart_test.rst"
        study_dir = "test"
        restarted_restart_file = os.path.join(study_dir,restart_filename)
        restart_results_file = os.path.join(study_dir, 
                                    IN_PROGRESS_RESULTS_FILENAME+"."+EVALUATION_EXTENSION)
        goal_restart = os.path.abspath(restarted_restart_file)
        initial_emod_val = emod_cur_val*(1+1e-2*np.random.uniform())
        def setup_cal(restart=False):
            EPC.update_parameters(elastic_modulus=initial_emod_val)
            calibration = GradientCalibrationStudy(EPC["elastic_modulus"])
            calibration.set_restart_filename(restart_filename)
            calibration.set_working_directory(study_dir)
            if restart:
                python_model.reset_constants()
                calibration.restart(restarted_restart_file, restart_results_file)
            calibration.add_evaluation_set(python_model, objective, goal_data)
            calibration.add_evaluation_set(model, curve_objective, goal_data)        
            calibration.set_core_limit(32)

            return calibration

        calibration = setup_cal()
        init_dir = os.getcwd()
        with self.assertRaises(AttributeError):
            #catches desired failure once calibration gets close to solution
            results = calibration.launch()
        import time
        matcal_print_message("Sleeping to allow files to delete")
        time.sleep(10.0)
        matcal_print_message("Going")
        os.chdir(init_dir)
        calibration = setup_cal(restart=True)
        results = calibration.launch()
        with open(os.path.join(study_dir, "dakota.out"), "r") as f:
            output = f.read()
        self.assertTrue(f"Reading restart file \'{goal_restart}\'." in output)
        self.assertTrue(f"Restart record    1" in output)

        goal_elastic_mod = MODEL_CONSTANTS["elastic_modulus"]
        self.assertAlmostEqual(results.outcome["best:elastic_modulus"], 
                               goal_elastic_mod, 
                               delta=goal_elastic_mod * self.error_tol)
        self.assertEqual(os.getcwd(), calibration._initial_directory)
        self.assertNotEqual(os.getcwd(), calibration._working_directory)
        self.assertTrue(os.path.exists(study_dir))
        self.assertTrue(os.path.exists(os.path.join(study_dir, "dakota.out")))
        self.assertTrue(os.path.exists(os.path.join(study_dir, MATCAL_TEMPLATE_DIRECTORY)))
        self.assertTrue(os.path.exists(calibration.final_results_filename))

