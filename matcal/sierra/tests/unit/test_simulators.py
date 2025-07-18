import numpy as np

from matcal.core.computing_platforms import (HPCComputingPlatform, 
    MatCalComputingPlatformFunctionIdentifier)
from matcal.core.data import DataCollection, convert_dictionary_to_data
from matcal.core.parameters import Parameter, ParameterCollection
from matcal.core.tests.MatcalUnitTest import MatcalUnitTest

from matcal.sierra.material import Material
from matcal.sierra.models import UniaxialLoadingMaterialPointModel, UserDefinedSierraModel
from matcal.sierra.tests.platform_options import MatCalTestPlatformOptionsFunctionIdentifier
from matcal.sierra.tests.utilities import (write_linear_elastic_material_file, 
    write_empty_file, write_design_param_file)


SET_PLATFORM_OPTIONS = MatCalTestPlatformOptionsFunctionIdentifier.identify()


class TestSierraSimulator(MatcalUnitTest):

    def setUp(self) -> None:
        super().setUp(__file__)
        self.empty_input_file = "empty.i"
        self.empty_mesh_file = "empty.g"
        write_empty_file(self.empty_input_file)
        write_empty_file(self.empty_mesh_file)
        write_design_param_file()
        
        data_1 = convert_dictionary_to_data({"engineering_strain":np.array([0, 0.1])})
        data_2 = convert_dictionary_to_data({"engineering_strain":np.array([0, 0.112])})
        self.state = data_1.state
        self.data_collection = DataCollection('my data', data_1, data_2)
        filename = write_linear_elastic_material_file()
        self.material_example = Material('matcal_test', filename,
                                        "linear_elastic")

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
        model = UniaxialLoadingMaterialPointModel(self.material_example)
        model.add_boundary_condition_data(self.data_collection)
        model.add_constants(nu=0.3, elastic_modulus=10e6, density=0.0025)
        SET_PLATFORM_OPTIONS(model)
        parameters = ParameterCollection("test", Parameter("a", 0,1))
        sim_results = model.run_check_syntax(self.state, parameters)
        self.assertEqual(sim_results.return_code,0)
        self._check_sierra_success_simulator(sim_results.stdout, sim_results.stderr)

    def test_create_sierra_simulator_check_input(self):
        model = UniaxialLoadingMaterialPointModel(self.material_example)
        model.add_boundary_condition_data(self.data_collection)
        model.add_constants(nu=0.3, elastic_modulus=10e6, density=0.0025)
        SET_PLATFORM_OPTIONS(model)
        parameters = ParameterCollection("test", Parameter("a", 0,1))
        sim_results = model.run_check_input(self.state, parameters)
        self.assertEqual(sim_results.return_code,0)

        self._check_sierra_success_simulator(sim_results.stdout, sim_results.stderr)

    def test_default_simulation_failure_flag(self):
        model = UniaxialLoadingMaterialPointModel(self.material_example)
        model.add_boundary_condition_data(self.data_collection)
        SET_PLATFORM_OPTIONS(model)
        model.preprocess(self.state)
        sim = model.build_simulator(self.state)
        self.assertTrue(sim.fail_calibration_on_simulation_failure)

    def test_continue_on_simulation_failure_flag(self):
        model = UniaxialLoadingMaterialPointModel(self.material_example)
        model.add_boundary_condition_data(self.data_collection)
        model.continue_when_simulation_fails()
        model.preprocess(self.state)
        sim = model.build_simulator(self.state)
        self.assertFalse(sim.fail_calibration_on_simulation_failure)

    def test_create_sierra_simulator_sprint_module(self):
        model = UniaxialLoadingMaterialPointModel(self.material_example)
        model.add_boundary_condition_data(self.data_collection)
        model.add_environment_module("sierra/sprint")
        model.add_constants(nu=0.3, elastic_modulus=10e6, density=0.0025)
        SET_PLATFORM_OPTIONS(model)
        model.preprocess(self.state)
        sim = model.build_simulator(self.state)
        parameters = {}
        sim_results = sim.run_check_syntax(parameters)
        self._check_sierra_success_simulator(sim_results.stdout, sim_results.stderr)

    def test_create_user_defined_simulator(self):
        model = UserDefinedSierraModel('adagio', self.empty_input_file, self.empty_mesh_file)
        SET_PLATFORM_OPTIONS(model)
        model.preprocess(self.state)
        model.continue_when_simulation_fails()
        sim = model.build_simulator(self.state)
        parameters = {}
        sim_results = sim.run_check_syntax(parameters)
        self._check_sierra_abort_simulator(sim_results.stdout, sim_results.stderr)

    def test_get_default_commands(self):
        model = UserDefinedSierraModel('aria', self.empty_input_file, self.empty_mesh_file)
        SET_PLATFORM_OPTIONS(model)
        sim = model.build_simulator(self.state)
        platform_identifier = MatCalComputingPlatformFunctionIdentifier.identify()
        platform = platform_identifier()
        if isinstance(platform, HPCComputingPlatform):
            queue_id = model.queue_id
            queues = platform.get_usable_queue_names(300, 1)
            if queues is not None:
                queues = ",".join(queues)
            goal_commands = ['sierra', '--run', '-n', '1', '--queue-name', queues, '-T', '1800',
                            '--account', queue_id, '--ppn', 
                            f'{platform.get_processors_per_node()}',
                            '-a', 'aria', '-i', 'empty.i', '--graceful_timeout', '1800']
        else:
            goal_commands = ['sierra', '--run', '-n', '1', '-a', 'aria', '-i', 'empty.i']

        commands = sim._commands       
        self._confirm_command_list(commands, goal_commands)

    def test_get_aria_default_time_out_commands(self):
        model = UserDefinedSierraModel('aria', self.empty_input_file, self.empty_mesh_file)
        sim = model.build_simulator(self.state)
        time_out_commands = sim._make_timeout_commands()
        self.assertIsNone(time_out_commands)

    def test_get_adagio_default_time_out_commands(self):
        model = UserDefinedSierraModel('adagio', self.empty_input_file, self.empty_mesh_file)
        sim = model.build_simulator(self.state)
        time_out_commands = sim._make_timeout_commands()
        self.assertIsNone(time_out_commands)

    def test_get_aria_time_out_commands(self):
        model = UserDefinedSierraModel('aria', self.empty_input_file, self.empty_mesh_file)
        model.set_time_limit(0.5)
        sim = model.build_simulator(self.state)
        goal = ["--graceful_timeout", "1800"]
        time_out_commands = sim._make_timeout_commands()
        self.assertListEqual(goal, time_out_commands)

    def test_get_custom_executable_arguments(self):
        exes = ['aria', 'adagio']
        for exe in exes:
            model = UserDefinedSierraModel(exe, self.empty_input_file, self.empty_mesh_file)
            model.add_executable_argument("--beta")
            sim = model.build_simulator(self.state)
            goal = ["--beta"]
            test = sim._make_custom_commands()
            self.assertListEqual(test, goal)

    def test_raises_not_string_error_for_custom_flags(self):
        entries = [1, None, [], ['asd'], {'name':3}]
        for value in entries:
            model = UserDefinedSierraModel('exe', self.empty_input_file, self.empty_mesh_file)
            with self.assertRaises(TypeError):
                model.add_executable_argument(value)

    def test_confirm_full_command_line(self):
        model = UserDefinedSierraModel('aria', self.empty_input_file, self.empty_mesh_file)
        model.set_time_limit(0.5)
        model.add_executable_argument("--beta")
        sim = model.build_simulator(self.state)
        goal = ['sierra', '--run', '-n', '1', '-a', 'aria', '-i', 'empty.i', 
                '--graceful_timeout', '1800', '--beta']
        self.assertListEqual(goal, sim._commands)

    def test_get_adagio_time_out_commands(self):
        model = UserDefinedSierraModel('adagio', self.empty_input_file, self.empty_mesh_file)
        model.set_time_limit(1)
        sim = model.build_simulator(self.state)
        time_out_commands = sim._make_timeout_commands()
        self.assertIsNone(time_out_commands)

    def _confirm_command_list(self, commands, goal_commands):
        self.assertEqual(len(goal_commands), len(commands))
        for goal, test in zip(goal_commands, commands):
            self.assertEqual(goal, test)


from copy import deepcopy
from matcal.core.linux_modules import (MatCalExecutableEnvironmentSetupFunctionIdentifier, 
                                       module_command_writer)

class TestSierraSimulatorShell(TestSierraSimulator):

    def setUp(self):
        super().setUp()
        self._orig_registry = deepcopy(MatCalExecutableEnvironmentSetupFunctionIdentifier._registry)
        self._orig_default = deepcopy(MatCalExecutableEnvironmentSetupFunctionIdentifier._default)
        MatCalExecutableEnvironmentSetupFunctionIdentifier._registry = {}
        MatCalExecutableEnvironmentSetupFunctionIdentifier.set_default(module_command_writer)

    def tearDown(self):
        super().tearDown()
        MatCalExecutableEnvironmentSetupFunctionIdentifier._registry = self._orig_registry
        MatCalExecutableEnvironmentSetupFunctionIdentifier.set_default(self._orig_default)