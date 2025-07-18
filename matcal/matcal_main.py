import argparse
import os

from matcal.core.logger import (initialize_matcal_logger, 
                                activate_debug_output, 
                                init_file_handler)
from matcal.core.object_factory import IdentifierByTestFunction


def get_arguments_local(argv, parser=None):
    if parser is None:
        parser = argparse.ArgumentParser(description="CEE version of MatCal. It runs your"
                                        "MatCal input deck on the CEE machine of your choice.", 
                                        conflict_handler='resolve')

    parser.add_argument("-d", "--debug", action="store_true", help="Turn on debug output "
                        "for increased logging output.")
    parser.add_argument("-i", "--input", type=str, help="MatCal input python script.")
 
    return parser.parse_args(argv)
 
    
def setup_and_get_input_path(argv=None):
    args = get_arguments_local(argv)
    logger = initialize_matcal_logger("matcal", add_stream_handlers=False)
    file_handler = init_file_handler()
    logger.addHandler(file_handler)
    if args.debug:
        activate_debug_output(file_handler)
    return os.path.abspath(args.input)


matcal_main_identifier = IdentifierByTestFunction(default=setup_and_get_input_path)