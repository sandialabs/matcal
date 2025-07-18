import logging
import os
import sys



from matcal.core.tests.MatcalUnitTest import MatcalUnitTest, capture_print
from matcal.core.logger import activate_debug_output, matcal_print_message, no_trace_back_exception_handler

def _my_fake_dakota_runner_output(logger):
    logger.info("dakota runner output")
    logger.debug("dakota runner debug")
    logger.error("dakota runner error")
    logger.warning("dakota runner warning")   

def _my_fake_dakota_interfaces_output(logger):
    logger.info("my analysis driver output")
    logger.debug("my analysis driver debug")
    logger.error("my analysis driver error")
    logger.warning("my analysis driver warning")

class MatCalFormaterTest(MatcalUnitTest):

    def setUp(self):
        super().setUp(__file__)

        from matcal.core.logger import MatcalFormatter
        self.formatter = MatcalFormatter()
        self.debug_record = logging.LogRecord("test", logging.DEBUG, "my_path.py", 45, "debug msg", (), None)
        self.warning_record = logging.LogRecord("test", logging.WARNING, "my_path.py", 45, "warning msg", (), None)
        self.info_record = logging.LogRecord("test", logging.INFO, "my_path.py", 45, "info msg", (), None)
        self.error_record = logging.LogRecord("test", logging.ERROR, "my_path.py", 45, "error msg", (), None)

    def test_format(self):
        debug_msg = self.formatter.format(self.debug_record)
        self.assertEqual(debug_msg, "DEBUG: debug msg FROM MODULE: test")

        info_msg = self.formatter.format(self.info_record)
        self.assertEqual(info_msg, "info msg")

        error_msg = self.formatter.format(self.error_record)
        self.assertEqual(error_msg, "ERROR: error msg")

        warning_msg = self.formatter.format(self.warning_record)
        self.assertEqual(warning_msg, "WARNING: warning msg")

    def test_combine_stack_filenames(self):
        stack_filenames, stack_functions = self.formatter._combine_stack_info()
        self.assertTrue("matcal/core/logger.py" in stack_filenames)
        self.assertTrue("matcal/core/tests/unit/test_logger.py" in stack_filenames)


class MatCalLoggerTest(MatcalUnitTest):

    def setUp(self):
        super().setUp(__file__)
        self.gold_with_debug = __file__.split("unit/logger/")[
                                   0] + "unit/logger/gold_log_files/logger_test_with_debug_gold.log"

    def tearDown(self):
        self._remove_handlers()
        super().tearDown()

    def _remove_handlers(self):
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)

    def _start_logger(self, tabs=0):
        from matcal.core.logger import (MatcalLogger, log_file_name, init_file_handler, 
            no_trace_back_exception_handler)
        self._log_file_name = log_file_name
        logging.setLoggerClass(MatcalLogger)
        self.logger = logging.getLogger(os.getcwd())
        local_file_handler = init_file_handler()
        self.logger.addHandler(local_file_handler)
        self._no_trace_back_exception_handler = no_trace_back_exception_handler
        return local_file_handler

    def test_standard_log(self):
        self._start_logger()
        self._write_some_output()
        _my_fake_dakota_runner_output(self.logger)
        _my_fake_dakota_interfaces_output(self.logger)

        gold_string = """my main output
ERROR: my main error
WARNING: my main warning
dakota runner output
ERROR: dakota runner error
WARNING: dakota runner warning
my analysis driver output
ERROR: my analysis driver error
WARNING: my analysis driver warning
"""
        self.assert_file_equals_string(gold_string, self._log_file_name)

    def test_disable_traceback_debug(self):
        self._start_logger()
        activate_debug_output()
        self.logger.disable_traceback()

        self.assertNotEqual(sys.excepthook, self._no_trace_back_exception_handler)
        self.assertEqual(sys.excepthook, sys.__excepthook__)

    def test_debug_log(self):
        file_handler = self._start_logger()
        activate_debug_output(file_handler)

        self._write_some_output()
        _my_fake_dakota_runner_output(self.logger)
        _my_fake_dakota_interfaces_output(self.logger)

        gold_string = """my main output
DEBUG: my main debug FROM MODULE: /gpfs1/knkarls/projects/matcal_working/matcal/test/unit/logger/test_logger_test_3
ERROR: my main error
WARNING: my main warning
dakota runner output
DEBUG: dakota runner debug FROM MODULE: /gpfs1/knkarls/projects/matcal_working/matcal/test/unit/logger/test_logger_test_3
ERROR: dakota runner error
WARNING: dakota runner warning
my analysis driver output
DEBUG: my analysis driver debug FROM MODULE: /gpfs1/knkarls/projects/matcal_working/matcal/test/unit/logger/test_logger_test_3
ERROR: my analysis driver error
WARNING: my analysis driver warning
"""
        with open(self._log_file_name) as lfid:
            log_string = lfid.read()

        for gold_line, file_line in zip(gold_string.split("\n"), log_string.split('\n')):
            self.assertEqual(gold_line.split("FROM MODULE")[0], file_line.split("FROM MODULE")[0])

    def _write_some_output(self):
        self.logger.info("my main output")
        self.logger.debug("my main debug")
        self.logger.error("my main error")
        self.logger.warning("my main warning")


class OutputHandlerTest(MatcalUnitTest):
    def setUp(self):
        super().setUp(__file__)

    def tearDown(self):
        super().tearDown()

    def test_output_handler_default(self):
        from matcal.core.logger import no_trace_back_exception_handler, activate_debug_output
        sys.excepthook = no_trace_back_exception_handler
        activate_debug_output()
        self.assertEqual(sys.excepthook, sys.__excepthook__)



@capture_print
def captured_matcal_print(*args_to_print):
    matcal_print_message(*args_to_print)

@capture_print
def captured_no_trace_exception_handler(exception_type, exception, traceback):
    no_trace_back_exception_handler(exception_type, exception, traceback)


class TestMatCalPrints(MatcalUnitTest):
    
    def setUp(self):
        super().setUp(__file__)
        
    def test_matcal_print(self):
        message = "this is a test message"
        goal_message = f"{message}\n"
        printed_message = captured_matcal_print(message)
        self.assertEqual(printed_message, goal_message)

    def test_matcal_print_compoents(self):
        message_parts = ['this',"is","a","number", 4]
        goal_message = ""
        for part in message_parts:
            if len(goal_message) > 0:
                goal_message += " "
            goal_message += str(part)
        goal_message += "\n"
    
        printed_message = captured_matcal_print(*message_parts)
        self.assertEqual(printed_message, goal_message)
        
    def test_no_trace_back(self):
        exception_type = TypeError
        error_message = "Not right type"
        exception = TypeError(error_message)
        traceback = "RANDOM STUFF HERE"
        goal_message = f"TypeError: {error_message}\n"
        printed_message = captured_no_trace_exception_handler(exception_type, exception, traceback)
        self.assertEqual(printed_message, goal_message)
        
        
        

        
        
            
        