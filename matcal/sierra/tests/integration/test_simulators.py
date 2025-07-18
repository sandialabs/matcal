from matcal.core.state import SolitaryState
from matcal.core.tests.MatcalUnitTest import MatcalUnitTest

from matcal.sierra.tests.platform_options import MatCalTestPlatformOptionsFunctionIdentifier
from matcal.sierra.tests.sierra_sm_models_for_tests import (
    UniaxialLoadingMaterialPointModelForTests, UserDefinedSierraModelForTests)
from matcal.sierra.tests.utilities import write_design_param_file


SET_PLATFORM_OPTIONS = MatCalTestPlatformOptionsFunctionIdentifier.identify()


class TestSierraSimulator(MatcalUnitTest):

    def setUp(self) -> None:
        super().setUp(__file__)
        write_design_param_file()
        
    def test_continue_on_simulation_failure_flag(self):
        model_creator = UniaxialLoadingMaterialPointModelForTests()
        model = model_creator.init_model()
        for bc_dc in model_creator.boundary_condition_data_sets:
            model.add_boundary_condition_data(bc_dc)
            model = SET_PLATFORM_OPTIONS(model)
            model.continue_when_simulation_fails()
            for state in bc_dc.states.values():
                model.preprocess(state)
                sim = model.build_simulator(state)
                self.assertFalse(sim.fail_calibration_on_simulation_failure)

    def _check_sierra_abort_simulator(self, stdout, stderr):
        aprepro_ran_cluster = "Running aprepro" in stdout
        aprepro_ran_local = "Executing: aprepro" in stdout
        self.assertTrue(aprepro_ran_cluster or aprepro_ran_local)

        sierra_submitted_cluster = "job" in stdout
        sierra_failed_local = "SIERRA ABORT" in stderr
        self.assertTrue(sierra_submitted_cluster or sierra_failed_local)

    def _check_sierra_success_simulator(self, stdout, stderr):
        aprepro_ran_cluster = "Running aprepro" in stdout
        aprepro_ran_local = "Executing: aprepro" in stdout
        self.assertTrue(aprepro_ran_cluster or aprepro_ran_local)

        sierra_submitted_cluster = "job" in stdout
        sierra_ran_local = "Executing sierra" in stdout
        self.assertTrue(sierra_submitted_cluster or sierra_ran_local)

    def test_create_sierra_simulator(self):
        uniax_mat_point_model_for_test = UniaxialLoadingMaterialPointModelForTests()
        model = uniax_mat_point_model_for_test.init_model()
        bc_dc = uniax_mat_point_model_for_test.boundary_condition_data_sets[0]
        model.add_boundary_condition_data(bc_dc)
        SET_PLATFORM_OPTIONS(model)

        for state in bc_dc.states.values():
            model.preprocess(state)
            sim = model.build_simulator(state)
            parameters = {}
            sim_results = sim.run_check_input(parameters)
            self._check_sierra_success_simulator(sim_results.stdout, sim_results.stderr)
            break

    def test_create_sierra_simulator_sprint_module(self):
        uniax_mat_point_model_for_test = UniaxialLoadingMaterialPointModelForTests()
        model = uniax_mat_point_model_for_test.init_model()
        model.add_environment_module("sierra/sprint")
        bc_dc = uniax_mat_point_model_for_test.boundary_condition_data_sets[0]
        model.add_boundary_condition_data(bc_dc)
        SET_PLATFORM_OPTIONS(model)

        for state in bc_dc.states.values():
            model.preprocess(state)
            sim = model.build_simulator(state)
            parameters = {}
            sim_results = sim.run_check_input(parameters)
            self._check_sierra_success_simulator(sim_results.stdout, sim_results.stderr)
            break

    def test_create_user_defined_simulator(self):
        
        model = UserDefinedSierraModelForTests().init_model()
        model.continue_when_simulation_fails()
        SET_PLATFORM_OPTIONS(model)
        model.preprocess(SolitaryState())
        sim = model.build_simulator(SolitaryState())
        parameters = {}
        sim_results = sim.run_check_syntax(parameters)
        self._check_sierra_abort_simulator(sim_results.stdout, sim_results.stderr)




