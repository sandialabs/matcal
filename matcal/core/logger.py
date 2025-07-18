import sys
import logging
import inspect

import matcal.core.constants as mc

def no_trace_back_exception_handler(exception_type, exception, traceback):
    print_message = "%s: %s" % (exception_type.__name__, exception)
    matcal_print_message(print_message)

def matcal_print_message(*print_message):
    print(*print_message)

log_file_name = mc.LOG_FILE_NAME
LOGGING_LEVEL = logging.DEBUG

class MatcalFormatter(logging.Formatter):

    _err_fmt  = "{label}ERROR: {{message}}"
    _warn_fmt = "{label}WARNING: {{message}}"

    _dbg_fmt  = "{label}DEBUG: {{message}} FROM MODULE: {{name}}"
    _info_fmt = "{label}{{message}}"

    FORMATS = {
        logging.ERROR:  _err_fmt,
        logging.WARNING: _warn_fmt,
        logging.INFO: _info_fmt,
        logging.DEBUG:  _dbg_fmt,
    }

    def __init__(self):
        self._style_fmt = "{"
        self._date_fmt = '%m/%d/%Y %I:%M:%S %process'
        super().__init__( self.FORMATS[logging.INFO], datefmt=self._date_fmt, style=self._style_fmt)

    def format(self, record):
        record.msg = record.msg.encode().decode("ascii")
        combined_stack_filenames, combined_stack_functions = self._combine_stack_info()
        level_format = self.FORMATS[record.levelno].format(label=self._determine_label(combined_stack_filenames,
                                                                                       combined_stack_functions))
        formatter = logging.Formatter(level_format, datefmt=self._date_fmt, style=self._style_fmt)
        return formatter.format(record)

    @staticmethod
    def _combine_stack_info():
        combined_stack_filenames = ""
        combined_stack_functions = ""

        for stack in inspect.stack():
            combined_stack_filenames += stack.filename
            combined_stack_functions += stack.function

        return combined_stack_filenames, combined_stack_functions

    @staticmethod
    def _determine_label(combined_stack_filenames, combined_stack_functions):
        label = ""

        return label


class MatcalLogger(logging.Logger):
    def __init__(self, name):
        super().__init__(name)
        self.setLevel(LOGGING_LEVEL)

    @staticmethod
    def disable_traceback():
        if  output_stream_handler.level != logging.DEBUG:
            sys.excepthook = no_trace_back_exception_handler

    @staticmethod
    def enable_traceback():
        sys.excepthook = sys.__excepthook__


def info_and_warning_filter(record):
    return record.levelno == logging.INFO or record.levelno == logging.WARNING


def debug_and_info_and_warning_filter(record):
    return record.levelno == logging.DEBUG or record.levelno == logging.WARNING or record.levelno == logging.INFO


def init_file_handler(filename=log_file_name):
    file_handler = logging.FileHandler(filename, "w", delay=True)
    file_handler.setFormatter(MatcalFormatter())
    file_handler.setLevel(logging.INFO)
    return file_handler


output_stream_handler = logging.StreamHandler(sys.stdout)
output_stream_handler.setFormatter(MatcalFormatter())
output_stream_handler.setLevel(logging.INFO)
output_stream_handler.addFilter(info_and_warning_filter)

error_stream_handler = logging.StreamHandler(sys.stderr)
error_stream_handler.setFormatter(MatcalFormatter())
error_stream_handler.setLevel(logging.ERROR)
error_stream_handler.addFilter(lambda record: record.levelno >= logging.ERROR)
sys.excepthook = sys.__excepthook__


def activate_debug_output(file_handler=None):
    output_stream_handler.setLevel(logging.DEBUG)
    output_stream_handler.removeFilter(info_and_warning_filter)
    output_stream_handler.addFilter(debug_and_info_and_warning_filter )
    sys.excepthook = sys.__excepthook__
    if file_handler is not None:
        file_handler.setLevel(logging.DEBUG)


def initialize_matcal_logger(name, add_stream_handlers=True):
    logging.setLoggerClass(MatcalLogger)
    logger = logging.getLogger(name)
    if not logger.handlers and add_stream_handlers:
        logger.addHandler(output_stream_handler)
        logger.addHandler(error_stream_handler)
    return logger
