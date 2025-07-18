# This is a wrapper class to the unittest TestCase class Such that it contains 
# many of the common desired features in the class.
import mmap
from filecmp import cmp, clear_cache
from shutil import rmtree
from unittest import TestCase
from itertools import count
from matcal.core.logger import matcal_print_message
from numpy.random import randint
import glob
import os
import numpy as np
import sys
import time
import io
from contextlib import redirect_stdout

from matcal.core.constants import MATCAL_WORKDIR_STR
from matcal.core.utilities import remove_directory

TIME_FILENAME = "test_timings.txt"
OLD_TIME_FILENAME = "test_timings_old.txt"

def setUpTiming():
    if os.path.exists(OLD_TIME_FILENAME):
        matcal_print_message(f"Removing old timing file: {OLD_TIME_FILENAME}")
        os.remove(OLD_TIME_FILENAME)
    if os.path.exists(TIME_FILENAME):
        matcal_print_message(f"Copying {TIME_FILENAME} to {OLD_TIME_FILENAME}")
        os.rename(TIME_FILENAME, OLD_TIME_FILENAME)

class MatcalUnitTest(TestCase):
    _build_dir_count = count(0)

    def setUp(self, filename):
        test_filename = self.get_test_filename(filename)
        matcal_print_message(f"\nTesting: {self.id()}")
        self._start_time = time.time()
        super().setUp()
        self._auto_remove = os.getenv("MATCAL_UNIT_TEST_REMOVE", "True").lower() == "true"
        self.maxDiff = None  # remove diff reporting limits
        self.initial_dir = os.getcwd()
        self.build_dir = self.create_buid_dir(self.initial_dir, test_filename)
        os.environ["IS_MATCAL_TESTING"] = "True"

        os.chdir(self.build_dir)

    def assert_error_type(self, error_type, call, *args, **kargs):
        with self.assertRaises(error_type):
            call(*args, **kargs)

    def tearDown(self):
        os.chdir(self.initial_dir)
        ok=True
        _end_time = time.time()
        elapsed_time = _end_time - self._start_time
        with open(TIME_FILENAME, 'a') as tf:
            line = f"{self.id()}:{elapsed_time}\n"
            tf.write(line)
        try:
            ok, result = self.has_test_failed()
        except:
            pass
        if self._auto_remove and ok:
            remove_directory(self.build_dir)

    def has_test_failed(self):
        if hasattr(self._outcome, 'errors'):
            # Python 3.4 - 3.10  (These two methods have no side effects)
            result = self.defaultTestResult()
            self._feedErrorsToResult(result, self._outcome.errors)
        else:
            # Python 3.11+
            result = self._outcome.result
        ok = all(test != self for test, text in result.errors + result.failures)

        return ok, result

    def create_buid_dir(self, test_location, test_filename):
        dir_path = os.path.join(test_location, "{}_test_{}".format(test_filename, 
                                                                   str(next(self._build_dir_count))
                                                                   ))
        remove_directory(dir_path)
        os.makedirs(dir_path)
        return dir_path

    def get_current_files_path(self, F):
        # call from a file with __file__ as F to get that file's abs path
        # used for setting paths to access local files
        path = os.path.dirname(F)
        path = os.path.abspath(path)
        if path == "":
            path = "."
        return (path)

    def get_test_filename(self, filename):
        filename = os.path.abspath(filename)
        filename = filename.split('/')[-1].split('.')[0]
        return filename

    def remove_file(self, file_path):
        if os.path.isfile(file_path):
            os.remove(file_path)

    def purge_matcal_workdirs(self):
        for mc_dir in glob.glob(MATCAL_WORKDIR_STR+".*"):
            rmtree(mc_dir)

    def create_temporary_dir(self, name="temp_dir"):
        os.makedirs(name)
        rel_path = "./{}".format(name)
        abs_path = os.path.abspath(rel_path)
        return (rel_path, abs_path)

    ## Resurn a list of the differences between two files
    def get_string_file_diff(self, filename_a, filename_b, space_replace='`'):

        diff_name = 'files.diff'

        os.system('diff ' + filename_a + " " + filename_b + "&> " + diff_name)

        f = open(diff_name, 'r')
        diff_note = ('\n< ' + filename_a + "\n>" + filename_b + '\n\n'
                      + "NOTE:: Spaces Replaced with \"{}\"\n".format(
                       space_replace))
        for line in f:
            diff_note += line.replace(" ", space_replace) + ''
        f.close()

        self.remove_file(diff_name)
        return (diff_note)

    def assert_same_string_file(self, filename_a, filename_b):
        clear_cache()
        result = cmp(filename_a, filename_b)
        file_diff = None
        if not result:
            file_diff = self.get_string_file_diff(filename_a, filename_b)
        self.assertTrue(result, msg=file_diff)

    def assert_write_indent(self, IF, goal, indent=1):
        IDX = randint(0, 1000000)
        goal_file = self.build_dir + "/goal_file_{}.txt".format(IDX)
        test_file = self.build_dir + "/test_file_{}.txt".format(IDX)
        f = open(test_file, 'w')
        IF.write(f, indent)
        f.close()
        g = open(goal_file, 'w')
        g.write(goal)
        g.close()

        self.assert_same_string_file(goal_file, test_file)
        self.remove_file(goal_file)
        self.remove_file(test_file)

    def assert_raise_error_on_input_block_write(self, block, error_type):
        IDX = randint(0, 1000000)
        test_file = self.build_dir + "/test_file_{}.txt".format(IDX)
        f = open(test_file, 'w')
        with self.assertRaises(error_type):
            block.write(f)
        f.close()
        self.remove_file(test_file)

    def assert_write(self, IF, goal):
        IDX = randint(0, 1000000)
        goal_file = self.build_dir + "/goal_file_{}.txt".format(IDX)
        test_file = self.build_dir + "/test_file_{}.txt".format(IDX)
        f = open(test_file, 'w')
        IF.write(f)
        f.close()
        g = open(goal_file, 'w')
        g.write(goal)
        g.close()

        self.assert_same_string_file(goal_file, test_file)
        self.remove_file(goal_file)
        self.remove_file(test_file)

    def assert_same_long_strings(self, test_string, goal_string):
        IDX = randint(0, 1000000)
        goal_file = self.build_dir + "/goal_file_{}.txt".format(IDX)
        test_file = self.build_dir + "/test_file_{}.txt".format(IDX)
        with open(test_file, 'w') as test_f:
            test_f.write(test_string)
        with open(goal_file, 'w') as goal_f:
            goal_f.write(goal_string)

        self.assert_same_string_file(goal_file, test_file)
        self.remove_file(goal_file)
        # self.remove_file(test_file)

    def assert_write_filename(self, IF, goal):
        IDX = randint(0, 1000000)
        goal_file = self.build_dir + "/goal_file_{}.txt".format(IDX)
        test_file = self.build_dir + "/test_file_{}.txt".format(IDX)
        IF.write(test_file)

        g = open(goal_file, 'w')
        g.write(goal)
        g.close()

        self.assert_same_string_file(goal_file, test_file)
        self.remove_file(goal_file)
        self.remove_file(test_file)

    def assert_file_equals_string(self, expected_str, filename):
        with open(filename) as fid:
            read_string = fid.read()
        self.assertEqual(expected_str, read_string)

    def assert_string_in_file(self, expected_str, filename):
        string_found = False
        with open(filename, 'rb', 0) as search_file, \
                mmap.mmap(search_file.fileno(), 0, access=mmap.ACCESS_READ) as s:
            if s.find(expected_str.encode('utf-8')) != -1:
                string_found = True
        self.assertTrue(string_found)

    def _write_gold_from_reference(self, goal_name, reference_name):
        self.remove_file(goal_name)
        with open(reference_name, 'r') as ref_f:
            with open(goal_name, "w") as goal_f:
                for ref_line in ref_f:
                    index = self._get_replacement_index(ref_line)
                    if index < 1:
                        goal_f.write(ref_line)
                    else:
                        self._write_replacement(goal_f, index)

    def check_if_close_arrays(self, first_array, second_array, atol=1e-08, rtol=1e-05, 
                              show_arrays=False, show_on_fail=False):
        if show_arrays:
            matcal_print_message('raw')
            matcal_print_message(first_array)
            matcal_print_message(second_array)
            matcal_print_message("delta")
            matcal_print_message(first_array - second_array)

        first_array = np.array(first_array)
        second_array = np.array(second_array)

        if first_array.dtype.names is not None:
            first_array = np.array(first_array.tolist())
        if second_array.dtype.names is not None:
            second_array = np.array(second_array.tolist())
        if len(first_array) == 0 and len(second_array) == 0:
            return True, np.array([]), np.array([])

        processed_first_array = first_array
        processed_first_array = first_array[np.isfinite(first_array)]
        if np.linalg.norm(first_array) < 1e-12:
            processed_first_array[0] += 1e-12
        processed_second_array = second_array
        processed_second_array = second_array[np.isfinite(second_array)]
        if np.linalg.norm(second_array) < 1e-12:
            processed_second_array[0] += 1e-12
        
        if show_arrays:
            matcal_print_message('processed')
            matcal_print_message(processed_first_array)
            matcal_print_message(processed_second_array)
            matcal_print_message("delta")
            delta_processed = processed_first_array-processed_second_array
            matcal_print_message(delta_processed)
            matcal_print_message("rel delta")
            matcal_print_message(np.divide(delta_processed, np.abs(processed_first_array)))
            matcal_print_message("rtol * abs(b)")
            matcal_print_message(rtol*np.abs(processed_second_array))
            matcal_print_message("a+rtol*abs(b)")
            matcal_print_message(atol+rtol*np.abs(processed_second_array))
            
        passed = np.allclose(processed_first_array, processed_second_array, atol=atol, rtol=rtol)
        return passed, processed_first_array, processed_second_array
    
    def assert_close_arrays(self, first_array, second_array, 
                            atol=1e-08, rtol=1e-05, show_arrays=False, show_on_fail=False):
        res = self.check_if_close_arrays(first_array, 
                                        second_array, 
                                        atol, rtol, 
                                        show_arrays, 
                                        show_on_fail)
        passed, processed_first_array, processed_second_array = res
        if not passed:
            delta = processed_first_array-processed_second_array
            matcal_print_message(f"Failed Error 2-Norm: {np.linalg.norm(delta)}"
                  f"  inf-Norm: {np.linalg.norm(delta, np.inf)}")
            if show_on_fail:
                matcal_print_message('processed')
                matcal_print_message(processed_first_array)
                matcal_print_message(processed_second_array)
                matcal_print_message("delta")
                delta_processed = processed_first_array-processed_second_array
                matcal_print_message(delta_processed)
                matcal_print_message("rel delta")
                matcal_print_message(np.divide(delta_processed, np.abs(processed_first_array))) 
                import matplotlib.pyplot as plt
                plt.plot(processed_first_array, '-x', label='1')
                plt.plot(processed_second_array, label='2')
                plt.legend()
                plt.show()
        self.assertTrue(passed)

    def assert_close_dicts_or_data(self, first, second, err_tol=1e-8, show_arrays=False):
        self.assertEqual(len(first.keys()), len(second.keys()))
        for key in first.keys():
            first_value = first[key]
            second_value = second[key]
            if show_arrays:
                matcal_print_message(key, type(first_value), type(second_value))
            if isinstance(first_value, str):
                self.assertEqual(first_value, second_value)
            elif isinstance(first_value, (int, float)):
                self.assertAlmostEqual(first_value, second_value)
            else:
                self.assert_close_arrays(first_value, second_value, atol=err_tol,
                                          show_arrays=show_arrays)

    def assert_list_contains_array(self, test_list, goal_array):
        passed = False
        for val in test_list:
            results = self.check_if_close_arrays(val, goal_array)
            if results[0]:
                passed  = True
                break
        self.assertTrue(passed)
        

    def _get_replacement_index(self, ref_line):
        if ref_line[0:3] == "#$#":
            return int(ref_line[3:7])
        else:
            return -1

    # to be overwritten as needed
    def _write_replacement(self, goal_f, index):
        pass

    def assert_file_exists(self, test_file):
        self.assertTrue(os.path.isfile(test_file))

    def assert_same_unordered_lists(self, a: list, b: list) -> None:
        self.assertIsInstance(a, list)
        self.assertIsInstance(b, list)
        if len(a) != len(b):
            matcal_print_message(f"BAD Lists:\n{a}\n{b}")
            self.assertTrue(False)
        okay = True
        for value in a:
            if value not in b:
                okay = False
        if not okay:
            matcal_print_message(f"BAD Lists:\n{a}\n{b}")
            self.assertTrue(False)


def is_linux():
    return sys.platform in ["linux", "linux2"]


def is_mac():
    return sys.platform == "darwin"

import argparse
import sys
def _set_up_arg_parser():
    parser = argparse.ArgumentParser(description="Postprocessing tool for MatCal tests", 
                                        conflict_handler='resolve')

    parser.add_argument("-t", "--timing", type=str, help="Process the timing file passed after flags", default=None)
    parser.add_argument("-c", "--clean", action='store_true', help="Remove all matcal test directories and files.")
    return parser


def _process_timing_file(time_filename):
    test_names, test_times = _parse_file(time_filename)
    test_names, test_times = _sort_by_time(test_names, test_times) 
    n_report = 10
    _print_longest_times(test_names, test_times, n_report)
    

def _sort_by_time(test_names, test_times):
    test_names = np.array(test_names)
    test_times = np.array(test_times)
    ascending_indices = np.argsort(test_times)
    test_times = test_times[ascending_indices]
    test_names = test_names[ascending_indices]
    return test_names,test_times

def _parse_file(time_filename):
    test_names = []
    test_times = []
    split_char = ":"
    with open(time_filename, 'r') as tf:
        for line in tf.readlines():
            name, time = line.split(split_char)
            test_names.append(name)
            test_times.append(float(time))
    return test_names,test_times   
    
def _print_longest_times(test_names, test_times, n_report):
    message = "Longest Running Tests:\n"
    for i_report in range(n_report):
        i_test = -(i_report + 1)
        message += f"  {test_names[i_test]}\n    {test_times[i_test]}\n"
    matcal_print_message(message)
        
def _clean_tests():
    command = "rm -rf test_*_test_*"
    os.system(command)

if __name__=="__main__":
    arg_parser = _set_up_arg_parser()
    passed_args = arg_parser.parse_args() 
    if passed_args.timing != None:
        _process_timing_file(passed_args.timing)
    if passed_args.clean:
        _clean_tests()
        
        

def capture_print(func):
    def wrapper(*args, **kwargs):
        f = io.StringIO()
        with redirect_stdout(f):
            func(*args, **kwargs)
        return f.getvalue()
    return wrapper
