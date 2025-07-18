from copy import deepcopy
import os

from matcal.core.computing_platforms import (local_computer, 
    MatCalComputingPlatformFunctionIdentifier, 
    LocalComputingPlatform)
from matcal.core.external_executable import (ListCommandError, 
    MatCalExternalExecutableFactory, MatCalExecutableEnvironmentSetupFunctionIdentifier, 
    default_environment_command_processor, ExecutableNoEnvironmentSetup, 
    MatCalPlatformEnvironmentSetupIdentifier, SlurmHPCExecutableEnvironmentSetup, 
    attempt_to_execute, NonPositiveIntegerError)
from matcal.core.tests.MatcalUnitTest import MatcalUnitTest


class TestLocalExternalExecutable(MatcalUnitTest):

    def setUp(self):
        super().setUp(__file__)
        self._purge_non_core_modules()

    def _purge_non_core_modules(self):
        self._orig_registry = deepcopy(MatCalExecutableEnvironmentSetupFunctionIdentifier._registry)
        self._orig_default = deepcopy(MatCalExecutableEnvironmentSetupFunctionIdentifier._default)
        MatCalExecutableEnvironmentSetupFunctionIdentifier._registry = {}
        MatCalExecutableEnvironmentSetupFunctionIdentifier.set_default(
            default_environment_command_processor)

    def tearDown(self) -> None:
        MatCalExecutableEnvironmentSetupFunctionIdentifier._registry = self._orig_registry 
        MatCalExecutableEnvironmentSetupFunctionIdentifier.set_default(self._orig_default)
        super().tearDown()

    def test_get_module_commands_back_for_general_environment_valid(self):
        my_env_command = 'export MYFAKEPATH=FAKEPATH'
        valid_runner = MatCalExternalExecutableFactory.create(['ls'], 
                                                              [my_env_command], 
                                                              local_computer)
        additional_commands, use_shell = valid_runner._setup_environment()
        self.assertEqual(my_env_command, additional_commands)

    def test_test_default_executable_uses_shell(self):
        my_env_command = 'export MYFAKEPATH=FAKEPATH'
        valid_runner = MatCalExternalExecutableFactory.create(['ls'], 
                                                              [my_env_command], 
                                                              local_computer)
        additional_commands, use_shell = valid_runner._setup_environment()
        self.assertTrue(use_shell)

    def test_get_multiple_module_commands_back_for_general_environment_valid(self):
        my_env_commands = ['export MYFAKEPATH=FAKEPATH', 'echo ISMATCALAWESOME']
        goal_command = 'export MYFAKEPATH=FAKEPATH;echo ISMATCALAWESOME'
        valid_runner = MatCalExternalExecutableFactory.create(['ls'],
                                                              my_env_commands, 
                                                              local_computer)
        additional_commands, use_shell = valid_runner._setup_environment()
        self.assertEqual(goal_command, additional_commands)

    def test_run_valid_commands(self):
        var_name = "MYMatcalTestVar"
        var_val = 23
        my_env_command = [f'export {var_name}={var_val}', f'echo ${var_name}']
        valid_runner = MatCalExternalExecutableFactory.create(['ls'], 
                                                              my_env_command, 
                                                              local_computer)
        stdout, stderr, process = valid_runner.run()
        self.assertEqual(len(stderr), 0)

    def test_run_invalid_commands(self):
        invalid_command_runner = MatCalExternalExecutableFactory.create(['not_a_VALID_command'],
                                                                         None,
                                                                        local_computer)

        stdout, stderr, process = invalid_command_runner.run()
        self.assertTrue(len(stderr) > 0)


class TestEnvSetupFactory(MatcalUnitTest):

    def setUp(self):
        super().setUp(__file__)
        MatCalExecutableEnvironmentSetupFunctionIdentifier._registry = {}

    def test_init(self):
        cef = MatCalExecutableEnvironmentSetupFunctionIdentifier
        self.assertIsNotNone(cef)

    def test_get_default(self):
        cef = MatCalExecutableEnvironmentSetupFunctionIdentifier
        proc = cef.identify()
        self.assertEqual(proc, default_environment_command_processor)


class TestDefaultEnvironmentCommand(MatcalUnitTest):

    def setUp(self):
        super().setUp(__file__)

    def test_take_in_one_command_returns_one_command(self):
        commands = ["export MYMATCALTESTVAR=4"]
        r_commands, use_shell = default_environment_command_processor(commands)
        self.assertEqual(commands[0], r_commands)

    def test_default_uses_shell(self):
        commands = ["export MYMATCALTESTVAR=4", "my_alias"]
        r_commands, use_shell = default_environment_command_processor(commands)
        self.assertTrue(use_shell)

    def test_take_in_commands_returns_commands(self):
        commands = ["export MYMATCALTESTVAR=4", "my_alias"]
        goal_commands = "export MYMATCALTESTVAR=4;my_alias"
        r_commands, use_shell = default_environment_command_processor(commands)
        self.assertTrue(use_shell)
        self.assertEqual(goal_commands, r_commands)

    def test_raise_error_not_list(self):
        cmd = 'export matcaltest=3'
        with self.assertRaises(ListCommandError):
            default_environment_command_processor(cmd)

    def test_identify_computing_platform(self):
        computing_platform_indentifier_func = \
            MatCalComputingPlatformFunctionIdentifier.identify()
        self.assertIsInstance(computing_platform_indentifier_func(), 
                              LocalComputingPlatform)
        

class TestExecutableEnvironmentIdentifier(MatcalUnitTest):

    def setUp(self):
        super().setUp(__file__)

    def test_default_returns_NoSetup(self):
        MatCalPlatformEnvironmentSetupIdentifier._registry = {}
        setup_class = MatCalPlatformEnvironmentSetupIdentifier.identify()
        self.assertIsInstance(setup_class, ExecutableNoEnvironmentSetup)    


class TestAttemptToExecute(MatcalUnitTest):

    def setUp(self):
        super().setUp(__file__)

    def test_execute_tool_with_no_inputs_correctly_returns_true(self):
        def simple_exe():
            pass
        max_attempts = 1
        pause_time = 0 
        passed, n_attempts = attempt_to_execute(simple_exe, max_attempts, pause_time)
        self.assertTrue(passed)
        self.assertEqual(n_attempts, 1)
        
    def test_execute_tool_with_args_returns_true(self):
        def simple_exe(*args):
            for a in args:
                pass
        max_attempts = 1
        pause_time = 0 
        args = ['1', 'a', 123]
        passed, n_attempts = attempt_to_execute(simple_exe, max_attempts, pause_time, *args)
        self.assertTrue(passed)
        self.assertEqual(n_attempts, 1)

    def test_execute_tool_with_kwargs_returns_true(self):
        def simple_exe(**kwargs):
            for k,v in kwargs.items():
                pass
        max_attempts = 1
        pause_time = 0 
        kwargs = {'a':12, 'b':123, 'c':1341234}
        passed, n_attempts = attempt_to_execute(simple_exe, max_attempts, pause_time, **kwargs)
        self.assertTrue(passed)
        self.assertEqual(n_attempts, 1)

    def test_execute_tool_with_args_kwargs_returns_true(self):
        def simple_exe(a, b, c=False, d=True):
            assert(a==1)
            assert(b==-2)
            assert(c)
            assert(not d)
            
        max_attempts = 1
        pause_time = 0 
        passed, n_attempts = attempt_to_execute(simple_exe, max_attempts, pause_time,
                                                 1, -2, d=False, c=True)
        self.assertTrue(passed)
        self.assertEqual(n_attempts, 1)

    def test_fail_n_times(self):
        class my_failure():
            def __init__(self, pass_after):
                self._pass_after = pass_after
                self._count = 1

            def pass_or_fail(self):
                if self._count > self._pass_after:
                    return True
                self._count += 1
                raise RuntimeError()

        self._confirm_success_after_iteration(my_failure, 2)
        self._confirm_success_after_iteration(my_failure, 1)
        self._confirm_success_after_iteration(my_failure, 0)
        self._confirm_success_after_iteration(my_failure, 10)
        self._confirm_success_after_iteration(my_failure, 7)

    def _confirm_success_after_iteration(self, my_failure, pass_after):
        fail_class = my_failure(pass_after)
        max_attempts = pass_after + 5
        goal_attempt = pass_after + 1
        pause_time = 0 
        passed, n_attempts = attempt_to_execute(fail_class.pass_or_fail, max_attempts, pause_time)
        self.assertTrue(passed)
        self.assertEqual(n_attempts, goal_attempt)
    
    def test_raise_error_if_less_than_1_attempt(self):
        def simple_exe():
            pass
        max_attempts = 0
        pause_time = 0 
        with self.assertRaises(NonPositiveIntegerError):
            passed, n_attempts = attempt_to_execute(simple_exe, max_attempts, pause_time)

        max_attempts = -1.1
        with self.assertRaises(NonPositiveIntegerError):
            passed, n_attempts = attempt_to_execute(simple_exe, max_attempts, pause_time)


class TestSlurmHPCEnvironmentSetup(MatcalUnitTest):
    def setUp(self):
        super().setUp(__file__)

    def test_slurm_environment_setup(self):
        os.environ["SLURM_JOB_ID"] = "test"
        os.environ["SLURM_JOBID"] = "test2"
        
        env_setup = SlurmHPCExecutableEnvironmentSetup()
        env_setup.prepare()
        self.assertTrue("SLURM_JOB_ID" not in os.environ.keys())
        self.assertTrue("SLURM_JOBID" not in os.environ.keys())

        env_setup.reset()
        self.assertEqual(os.environ["SLURM_JOB_ID"], "test")
        self.assertEqual(os.environ["SLURM_JOBID"], "test2")
        env_setup.prepare()