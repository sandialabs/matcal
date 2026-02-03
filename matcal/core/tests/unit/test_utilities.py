
import numpy as np
import os

from matcal.core.utilities import (_convert_list_of_files_to_abs_path_list, 
                                   _get_highest_version_subfolder, 
                                   is_text_file)
from matcal.core.utilities import (matcal_name_format, check_valid_matcal_name_string, 
  MatCalTypeStringError, MatcalNameFormatError, set_significant_figures, 
  make_clean_dir, get_current_time_string, check_item_is_correct_type, 
  check_value_is_positive, check_value_is_positive_integer, 
  check_value_is_positive_real, check_value_is_real_between_values, 
  check_value_is_nonempty_str, check_value_is_nonnegative_integer, 
  check_value_is_nonnegative_real)
from matcal.core.tests.MatcalUnitTest import MatcalUnitTest


class TestUtilities(MatcalUnitTest):

    def setUp(self) -> None:
        super().setUp(__file__)

    def test_convert_list_of_files_to_abs_path_list(self):
        file_1 = "test.txt"
        file_2 = "test2.txt"
    
        folder = "sub_dir"
        os.mkdir(folder)
        files = [file_1, file_2]
        for file in files:
            with open(file, "w") as f:
               f.write("\n")
        files.append(folder)

        abs_path_list = _convert_list_of_files_to_abs_path_list(files)
        for idx, file in enumerate(files):
            self.assertEqual(abs_path_list[idx], os.path.abspath(file))

    def test_get_highest_version_subfolder(self):
        os.mkdir("test_1.2.3")
        os.mkdir("test_2.4.3")
        os.mkdir("test_6.3.100")
        os.mkdir("test_6.5.3")

        highest_version_folder = _get_highest_version_subfolder(os.path.abspath("."))

        self.assertEqual(highest_version_folder, os.path.abspath("test_6.5.3"))

    def test_is_text_file(self):
        text_fname = "test.txt"
        with open(text_fname, "w") as f:
            f.write("\n")
        self.assertTrue(is_text_file(text_fname))

        nontext_fname = "text.bin"
        with open(nontext_fname, "wb") as f:
            os.urandom(1024)
        self.assertFalse(is_text_file(nontext_fname))

        folder_name = "subfolder"
        os.mkdir(folder_name)
        self.assertFalse(is_text_file(folder_name))

class TestCheckValidMatCalNameString(MatcalUnitTest):

  def setUp(self):
    super().setUp(__file__)

  def test_check_invalid_matcal_name_string(self):
    self.assert_error_type(MatCalTypeStringError, check_valid_matcal_name_string, "string_with_/")


  def test_check_valid_matcal_name_string(self):
    name = check_valid_matcal_name_string("valid")
    self.assertEqual(name, "valid")


class MatCalNameFormatTest(MatcalUnitTest):

  def setUp(self):
    super().setUp(__file__)
  
  def test_conversion(self):
  
    s = matcal_name_format("ALLCAPS")
    self.assertEqual(s,'ALLCAPS')
    
    s = matcal_name_format("spaces spaces")
    self.assertEqual(s,"spaces_spaces")
    s = matcal_name_format("Caps And SPACES")
    self.assertEqual(s,"Caps_And_SPACES")
    s = matcal_name_format("Mix_of Everything")
    self.assertEqual(s,"Mix_of_Everything")
    s = matcal_name_format("no_change_here")
    self.assertEqual(s,"no_change_here")
    
  def test_passList(self):
    list_of_strings = ["ALLCAPS","spaces Spaces","no_change"]
    result = matcal_name_format(list_of_strings)
    self.assertEqual(result[0],"ALLCAPS")
    self.assertEqual(result[1],"spaces_Spaces")
    self.assertEqual(result[2],"no_change")
    
  def test_errorCatch(self):
    self.assert_error_type(MatcalNameFormatError, matcal_name_format, 1)
    self.assert_error_type(MatcalNameFormatError, matcal_name_format,{})
    self.assert_error_type(MatcalNameFormatError, matcal_name_format,[])
    self.assert_error_type(MatcalNameFormatError, matcal_name_format,"")
    self.assert_error_type(MatcalNameFormatError, matcal_name_format,[1,2])
    self.assert_error_type(MatcalNameFormatError, matcal_name_format,None)

class TestSetSignificantFigures(MatcalUnitTest):
  def setUp(self):
    super().setUp(__file__)
    self.test_inputs = [
      1.114,  # positive, round down
      1.115,  # positive, round up
      -1.114,  # negative
      1.114e-16,  # extremely small
      1.114e16,  # extremely large
      0,  # zero
      2.112,
      float('inf'),  # infinite
    ]

    self.test_inputs_array_like = [[1.114, 1.115e-16], np.array([1.115, 1.114e-16])]
  def test_set_significant_figures_two(self):
    solutions = [
      1.1,  # positive, round down
      1.1,  # positive, round up
      -1.1,  # negative
      1.1e-16,  # extremely small
      1.1e16,  # extremely large
      0,  # zero
      2.1,
      float('inf'),  # infinite
      ]

    test_results = []
    for x in self.test_inputs:
       test_results.append(set_significant_figures(x, 2))

    for test_result, solution in zip(test_results,solutions):
      self.assertEqual(test_result, solution)

  def test_set_significant_figures_three(self):
    solutions = [
      1.11,  # positive, round down
      1.12,  # positive, round up
      -1.11,  # negative
      1.11e-16,  # extremely small
      1.11e16,  # extremely large
      0,  # zeros
      2.11,
      float('inf'),  # infinite
      ]

    test_results = []
    for x in self.test_inputs:
       test_results.append(set_significant_figures(x, 3))

    for test_result, solution in zip(test_results,solutions):
      self.assertEqual(test_result, solution)

  def test_array_like_set_significant_figures_three(self):
    solutions =  [[1.11, 1.12e-16], np.array([1.12, 1.11e-16])]

    test_results = []
    for x in self.test_inputs_array_like:
       test_results.append(set_significant_figures(x, 3))

    for test_result, solution in zip(test_results,solutions):
      for result_entry, solution_entry in zip(test_result, solution):
        self.assertEqual(result_entry, solution_entry)

class TestBasicUtilities(MatcalUnitTest):
  
    def setUp(self):
        super().setUp(__file__)

    def test_make_clean_dir(self):
        import os
        os.mkdir("test")
        os.mkdir("test/test_nested")
        make_clean_dir("test")
        self.assertTrue(os.path.exists("test"))

    def test_get_current_time_string(self):
        cur_time_str = get_current_time_string()
        self.assertIsInstance(cur_time_str, str)
        self.assertEqual(len(cur_time_str.split(":")), 3)
        self.assertEqual(len(cur_time_str.split("-")), 3)

    def test_check_item_is_correct_type(self):
        with self.assertRaises(TypeError):
            check_item_is_correct_type(1, str, "test")

    def test_check_value_is_positive(self):
        with self.assertRaises(ValueError):
            check_value_is_positive(-0.5, "test")

    def test_check_value_is_positive_real(self):
        with self.assertRaises(ValueError):
            check_value_is_positive_real(-0.5, "test")
        with self.assertRaises(ValueError):
            check_value_is_positive_real(0.0, "test")
        self.assertTrue(check_value_is_positive_real(1.0, "test"))

    def test_check_value_is_positive_integer(self):
        with self.assertRaises(TypeError):
            check_value_is_positive_integer(-0.5, "test")
        with self.assertRaises(ValueError):
            check_value_is_positive_integer(-5, "test")
        with self.assertRaises(ValueError):
            check_value_is_positive_integer(0, "test")
        self.assertTrue(check_value_is_positive_integer(1, "test"))

    def test_check_value_is_nonnegative_real(self):
        with self.assertRaises(ValueError):
            check_value_is_nonnegative_real(-0.5, "test")
        self.assertTrue(check_value_is_nonnegative_real(0.0, "test"))
        self.assertTrue(check_value_is_nonnegative_real(1.0, "test"))

    def test_check_value_is_nonnegative_integer(self):
        with self.assertRaises(TypeError):
            check_value_is_nonnegative_integer(-0.5, "test")
        with self.assertRaises(ValueError):
            check_value_is_nonnegative_integer(-5, "test")
        self.assertTrue(check_value_is_nonnegative_integer(0, "test"))
        self.assertTrue(check_value_is_nonnegative_integer(1, "test"))

    def test_check_value_is_real_between_values(self):
        with self.assertRaises(ValueError):
            check_value_is_real_between_values(1, 1, 2, "test", )
        check_value_is_real_between_values(1, 1, 2, "test",  True)
        with self.assertRaises(ValueError):
            check_value_is_real_between_values(0.5, 1, 2, "test", True)

    def test_check_value_is_nonempty_str(self):
        with self.assertRaises(TypeError):
            check_value_is_nonempty_str(1, "test")
        with self.assertRaises(ValueError):
            check_value_is_nonempty_str("", "test")
        #used to verify reporting of the error is correct.
        #I want to see 'test_func' 
        def test_func(my_param=1):
            check_value_is_nonempty_str(my_param, "test")
            
        with self.assertRaises(TypeError):
            test_func()
