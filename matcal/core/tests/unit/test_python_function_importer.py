import os

from matcal.core.tests.MatcalUnitTest import MatcalUnitTest
from matcal.core.python_function_importer import python_function_importer, PythonFunctionImportInputError, \
    PythonFunctionImporter, PythonLocalFunctionImporter, _picklable


def linear_python_model(**variables):
    time_max = 10
    num_time_steps = 100

    import numpy as np
    time = np.linspace(0, time_max, num_time_steps)
    values = variables['slope'] * time + variables['intercept']
    return {'time': time, "Y": values}


class TestPythonFunctionImporter(MatcalUnitTest):

    def setUp(self) -> None:
        def local_model(**variables):
            return linear_python_model(**variables)

        super().setUp(__file__)
        self.py_func_importer_global = python_function_importer(linear_python_model)
        self.py_func_importer_nonglobal = python_function_importer(local_model)

        self.py_func_file = os.path.join(self.get_current_files_path(__file__),
                                         "test_reference", "python_function_importer",
                                         "linear_python_model.py")
        self.py_func_importer_from_file = python_function_importer("linear_python_model", self.py_func_file)

    def test_try_to_pickle(self):
        def local_func(x):
            return x
        
        self.assertFalse(_picklable(local_func))
        self.assertTrue(_picklable(linear_python_model))
        

    def test_bad_init(self):
        with self.assertRaises(PythonFunctionImportInputError):
            python_function_importer(1, 1)

        with self.assertRaises(PythonFunctionImportInputError):
            python_function_importer("str", 1)

    def test_func_not_in_file(self):
        with self.assertRaises(PythonFunctionImporter.FunctionNotFound):
            py_func_importer = python_function_importer("not_in_file", self.py_func_file)

    def test_no_file(self):
        with self.assertRaises(PythonFunctionImporter.FileNotFound):
            py_func_importer = python_function_importer("func", "not a file")

    def test_get_function_import_path(self):
        goal_path = os.path.join(os.getcwd(), self.py_func_importer_nonglobal._imports_folder_base)
        self.assertEqual(self.py_func_importer_nonglobal.get_import_path(),goal_path)

    def test_return_python_function(self):
        func = self.py_func_importer_global.python_function
        vars = {'slope': 2, 'intercept': 1}
        values = func(**vars)

        import numpy as np
        values_goal = 2 * np.linspace(0, 10, 100) + 1.
        self.assert_close_arrays(values["Y"], values_goal)

    def test_return_python_function_from_file(self):
        func = self.py_func_importer_global.python_function
        vars = {'slope': 2, 'intercept': 1}
        values = func(**vars)

        import numpy as np
        values_goal = 2 * np.linspace(0, 10, 100) + 1.
        self.assert_close_arrays(values["Y"], values_goal)

    def test_pymodel_nonglobal_function(self):
        def linear_model_nonglobal(**variables):
            time_max = 10
            num_time_steps = 100

            import numpy as np
            time = np.linspace(0, time_max, num_time_steps)
            values = variables['slope'] * time + variables['intercept']
            return {'time': time, "Y": values}

        py_func = python_function_importer(linear_model_nonglobal).python_function

        func = py_func
        vars = {'slope': 2, 'intercept': 1}
        values = func(**vars)

        import numpy as np
        values_goal = 2 * np.linspace(0, 10, 100) + 1.
        self.assert_close_arrays(values["Y"], values_goal)

    def test_remove_string_leading_spaces(self):
        test_string_multiline = """  line 1
            line2
              line3
            line4
          line5
              line6"""
        goal_string_multiline = """line 1
          line2
            line3
          line4
        line5
            line6"""

        result_string = PythonLocalFunctionImporter.remove_function_leading_white_space(test_string_multiline)
        self.assertEqual(goal_string_multiline, result_string)

        test_string_one_line = "  line1"
        goal_string_one_line = "line1"
        result_string = PythonLocalFunctionImporter.remove_function_leading_white_space(test_string_one_line)
        self.assertEqual(goal_string_one_line, result_string)

        test_do_nothing_line = "do_nothing"
        result_string = PythonLocalFunctionImporter.remove_function_leading_white_space(test_do_nothing_line)
        self.assertEqual(test_do_nothing_line, result_string)

        test_do_nothing_mulitline = """do nothing
          do nothing again
            do nothing still
        nothing to do..."""
        
        result_string = PythonLocalFunctionImporter.remove_function_leading_white_space(test_do_nothing_mulitline)
        self.assertEqual(test_do_nothing_mulitline, result_string)

