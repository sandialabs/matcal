from multiprocessing import cpu_count
from socket import gethostname

from matcal.core.computing_platforms import (LocalComputingPlatform, 
    MatCalComputingPlatformFunctionIdentifier, MatCalJobDispatchDelayFunctionIdentifier,
    MatCalPermissionsCheckerFunctionIdentifier, no_check_checker, 
    JobSubmitCommandCreatorInterface, Queue, _convert_wall_time_string_to_seconds, 
    ImproperTimeFormatError)
from matcal.core.tests.MatcalUnitTest import MatcalUnitTest




class TestConvertWallTimeStringToSeconds(MatcalUnitTest):

    def setUp(self):
        super().setUp(__file__)

    def test_convert_wall_time_string_to_seconds(self):
        result = _convert_wall_time_string_to_seconds("60")
        self.assertEqual(result, 60)
        result = _convert_wall_time_string_to_seconds("3:25")
        self.assertEqual(result, 60*3+25)
        result = _convert_wall_time_string_to_seconds("1:2:35")
        self.assertEqual(result, 1*60*60+60*2+35)
        result = _convert_wall_time_string_to_seconds("2:3:1:45")
        self.assertEqual(result, 2*24*60*60+3*60*60+60*1+45)

    def test_convert_wall_time_string_to_seconds_invalid_input(self):
        with self.assertRaises(ImproperTimeFormatError):
            result = _convert_wall_time_string_to_seconds("uhoh")
        with self.assertRaises(ImproperTimeFormatError):
            result = _convert_wall_time_string_to_seconds("1:2:3:4:5")
        with self.assertRaises(ImproperTimeFormatError):
            result = _convert_wall_time_string_to_seconds(1)


class TestComputingPlatformFunctionIdentifier(MatcalUnitTest):

    def setUp(self):
        super().setUp(__file__)


    def test_default_returns_LocalComputerPlatform(self):
        MatCalComputingPlatformFunctionIdentifier._registry = {}
        platform_computing_identification_func = \
            MatCalComputingPlatformFunctionIdentifier.identify()
        comp = platform_computing_identification_func()
        self.assertIsInstance(comp, LocalComputingPlatform)


class TestDispatchDelayFunctionIdentifier(MatcalUnitTest):

    def setUp(self):
        super().setUp(__file__)

    def test_return_defualt_zero(self):
        MatCalJobDispatchDelayFunctionIdentifier._registry = {}
        delay_value_func = MatCalJobDispatchDelayFunctionIdentifier.identify()
        delay_value = delay_value_func()
        self.assertEqual(delay_value, 0)


class TestPermissionsChecker(MatcalUnitTest):

    def setUp(self):
        super().setUp(__file__)

    def test_core_returns_no_check_checker(self):
        MatCalPermissionsCheckerFunctionIdentifier._registry = {}
        check_func = MatCalPermissionsCheckerFunctionIdentifier.identify()
        self.assertEqual(check_func, no_check_checker)

class TestJobSubmitCommandCreatorInterface(MatcalUnitTest):

    def setUp(self):
        super().setUp(__file__)

    def test_job_submit_command_interface(self):
        goal_cmds = ["command", "arg"]
        goal_env_cmds = ["env_command", "env_arg"]
        job_commands = JobSubmitCommandCreatorInterface(goal_cmds, 
                                                        goal_env_cmds)
        cmds = job_commands.get_commands()
        self.assertEqual(cmds, goal_cmds)        
        env_cmds = job_commands.get_environment_setup_commands()
        self.assertEqual(env_cmds, goal_env_cmds)

class TestQueue(MatcalUnitTest):

    def setUp(self):
        super().setUp(__file__)

    def test_queue(self):
        q = Queue("dr queue", 10, 20)
        self.assertEqual(q.name, "dr queue")
        self.assertEqual(q.time_limit_seconds, 20)
        self.assertEqual(q.node_limit, 10)

class TestLocalComputingPlatform(MatcalUnitTest):

    def setUp(self):
        super().setUp(__file__)

    def test_queue(self):

        cp = LocalComputingPlatform("test", 8, 8, None)
        self.assertEqual(cp.name, gethostname())
        self.assertEqual(cp.total_processors, cpu_count())
        self.assertEqual(cp.processors_per_node, cpu_count())
        self.assertEqual(cp.get_usable_queue_names(100, 1), None)
        self.assertEqual(cp.queues, None)