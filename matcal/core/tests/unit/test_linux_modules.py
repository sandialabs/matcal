from copy import deepcopy

from matcal.core.computing_platforms import local_computer
import matcal.core.constants as matcal_const
from matcal.core.external_executable import MatCalExternalExecutableFactory
from matcal.core.linux_modules import (get_all_loaded_modules, 
                                         issue_module_commands, 
                                         default_modules_command_does_not_exist, 
                                         module_command_writer, module_command_executer)
from matcal.core.linux_modules import (MatCalTestModuleIdentifier, 
                                       MatCalExecutableEnvironmentSetupFunctionIdentifier)
from matcal.core.tests.MatcalUnitTest import MatcalUnitTest

import unittest


NO_MODULE_COMMANDS = default_modules_command_does_not_exist()


class TestLinuxModulesBase():
    def __init__(self):
        pass

    class CommonTests(MatcalUnitTest):
        def setUp(self):
            self._init_loaded_modules = get_all_loaded_modules()
            super().setUp(__file__)
            matcal_const.MODULE_PAUSE_TIME = .1
            self._test_module = MatCalTestModuleIdentifier.identify()

        def tearDown(self) -> None:
            issue_module_commands("purge")
            for module_name in self._init_loaded_modules:
                issue_module_commands("load", module_name)
            super().tearDown()

        @unittest.skipIf(NO_MODULE_COMMANDS, "Module commands do not exist")
        def test_load_valid_modules(self):
            issue_module_commands('purge')
            out, err = issue_module_commands('load', self._test_module)
            self.assertGreater(len(out), 0)

        @unittest.skipIf(NO_MODULE_COMMANDS, "Module commands do not exist")
        def test_load_invalid_modules(self):
            with self.assertRaises(RuntimeError):
                out, err = issue_module_commands('load', 'Not_A_Module')

        @unittest.skipIf(NO_MODULE_COMMANDS, "Module commands do not exist")
        def test_module_purge(self):
            out, err = issue_module_commands('purge')
            self.assertEqual(len(err), 0)

        @unittest.skipIf(NO_MODULE_COMMANDS, "Module commands do not exist")
        def test_loaded_modules(self):
            issue_module_commands("purge")
            issue_module_commands("load", self._test_module)
            loaded_modules = get_all_loaded_modules()
            test_module_found = False
            for module_name in loaded_modules:
                if self._test_module in module_name:
                    test_module_found = True
            self.assertTrue(test_module_found)


class TestLinuxModulesSubprocess(TestLinuxModulesBase.CommonTests):
    def setUp(self):
        super().setUp()


class TestLocalExternalExecutableModulesSubprocess(MatcalUnitTest):
    def setUp(self):
        super().setUp(__file__)
        self._test_module = MatCalTestModuleIdentifier.identify()
        self._orig_registry = deepcopy(MatCalExecutableEnvironmentSetupFunctionIdentifier._registry)
        self._orig_default = deepcopy(MatCalExecutableEnvironmentSetupFunctionIdentifier._default)
        MatCalExecutableEnvironmentSetupFunctionIdentifier._registry = {}
        MatCalExecutableEnvironmentSetupFunctionIdentifier.set_default(module_command_executer)

    def tearDown(self):
        super().tearDown()
        MatCalExecutableEnvironmentSetupFunctionIdentifier._registry = self._orig_registry
        MatCalExecutableEnvironmentSetupFunctionIdentifier.set_default(self._orig_default)
    
    @unittest.skipIf(NO_MODULE_COMMANDS, "Module commands do not exist")
    def test_setup_environment_valid(self):
        valid_runner = MatCalExternalExecutableFactory.create(['ls'], 
            [self._test_module], local_computer)
        additional_commands, use_shell = valid_runner._setup_environment()
        self.assertFalse(use_shell)
        self.assertEqual(additional_commands, [])

    @unittest.skipIf(NO_MODULE_COMMANDS, "Module commands do not exist")
    def test_setup_environment_invalid(self):
        invalid_runner = MatCalExternalExecutableFactory.create(['not_a_VALID_command'], 
            ['not_A_valid_Module'], local_computer)
        with self.assertRaises(RuntimeError):
            stdout, stderr, process = invalid_runner.run()

    @unittest.skipIf(NO_MODULE_COMMANDS, "Module commands do not exist")
    def test_run_valid_commands(self):
        valid_runner = MatCalExternalExecutableFactory.create(['ls'], [self._test_module], 
            local_computer)
        stdout, stderr, process = valid_runner.run()
        self.assertEqual(len(stderr), 0)
        self.assertEqual(process,  0)

    @unittest.skipIf(NO_MODULE_COMMANDS, "Module commands do not exist")
    def test_run_invalid_commands(self):
        invalid_command_runner = MatCalExternalExecutableFactory.create(['not_a_VALID_command'],
            None, local_computer)
        with self.assertRaises((FileNotFoundError,PermissionError)):
            stdout, stderr, process = invalid_command_runner.run()


class TestLocalExternalExecutableModulesShell(MatcalUnitTest):

    def setUp(self):
        super().setUp(__file__)
        self._test_module = MatCalTestModuleIdentifier.identify()
        self._orig_registry = deepcopy(MatCalExecutableEnvironmentSetupFunctionIdentifier._registry)
        self._orig_default = deepcopy(MatCalExecutableEnvironmentSetupFunctionIdentifier._default)
        MatCalExecutableEnvironmentSetupFunctionIdentifier._registry = {}
        MatCalExecutableEnvironmentSetupFunctionIdentifier.set_default(module_command_writer)

    def tearDown(self):
        super().tearDown()
        MatCalExecutableEnvironmentSetupFunctionIdentifier._registry = self._orig_registry
        MatCalExecutableEnvironmentSetupFunctionIdentifier.set_default(self._orig_default)

    @unittest.skipIf(NO_MODULE_COMMANDS, "Module commands do not exist")
    def test_setup_environment_valid(self):
        valid_runner = MatCalExternalExecutableFactory.create(['ls'], 
            [ self._test_module], local_computer)
        additional_commands, use_shell = valid_runner._setup_environment()
        goal_commands = "module purge;module load sierra;"
        self.assertTrue(use_shell)
        self.assertEqual(additional_commands, goal_commands)

    @unittest.skipIf(NO_MODULE_COMMANDS, "Module commands do not exist")
    def test_setup_environment_invalid(self):
        invalid_runner = MatCalExternalExecutableFactory.create(['not_a_VALID_command'], 
            ['not_A_valid_Module'], local_computer)
        stdout, stderr, process = invalid_runner.run()
        self.assertNotEqual(process,  0)

    @unittest.skipIf(NO_MODULE_COMMANDS, "Module commands do not exist")
    def test_run_valid_commands(self):
        valid_runner = MatCalExternalExecutableFactory.create(['ls'], [ self._test_module], local_computer)
        stdout, stderr, process = valid_runner.run()
        self.assertEqual(process, 0)

    @unittest.skipIf(NO_MODULE_COMMANDS, "Module commands do not exist")
    def test_run_invalid_commands(self):
        invalid_command_runner = MatCalExternalExecutableFactory.create(['not_a_VALID_command'],
            None, local_computer)
        stdout, stderr, process = invalid_command_runner.run()
        self.assertNotEqual(process,  0)

  
