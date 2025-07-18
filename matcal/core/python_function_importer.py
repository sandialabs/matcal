import inspect
from types import FunctionType
from importlib import util
import os

from matcal.core.surrogates import _MatCalSurrogateWrapper


class PythonFunctionImportInputError(RuntimeError):
    pass


def python_function_importer(python_function, filename=None):
        
        if isinstance(python_function, (FunctionType, _MatCalSurrogateWrapper)) and filename is None:
            picklable_function = _picklable(python_function)
            if picklable_function:
                return PythonPicklableFunctionImporter(python_function)
            else:
                py_func_import = PythonLocalFunctionImporter(python_function)
                return py_func_import
        elif isinstance(python_function, str) and isinstance(filename, str):
            return PythonFunctionImporter(python_function, filename)
        else:
            raise PythonFunctionImportInputError("An object using the python function importer can accept only two "
                                                 "types of input:\n\t(1) a funciton name defined locally in the file "
                                                 "\n\t (2) a function name defined in a separate file and the "
                                                 "full path filename where it is defined. \n The function name \"{}\" "
                                                 "and filename \"{}\" passed is invalid".format(python_function,
                                                                                                filename))

def _picklable(func):
    import pickle
    try:
        pickle.dumps(func)    
        return True
    except (AttributeError, pickle.PicklingError):
        return False


class PythonFunctionImporter:
    _module_name = "python_function_import_module"

    class FunctionNotFound(RuntimeError):
        pass

    class FileNotFound(RuntimeError):
        pass

    def __init__(self, python_function, filename):
        self._func_name = python_function
        self._check_file_exists(filename)
        self._filename = os.path.abspath(filename)
        self._check_function_can_be_imported()

    def _check_function_can_be_imported(self):
        try:
            self.python_function
        except AttributeError:
            raise self.FunctionNotFound("\n\nThe function \"{}\" was not found in file \"{}\"".format(self._func_name,
                                                                                                  self._filename))

    def _check_file_exists(self, filename):
        if not os.path.exists(filename):
            raise self.FileNotFound("\n\nThe file \"{}\" containing function \"{}\" "
                                    "does not exist.".format(filename, self._func_name))
    @property
    def python_function(self):
        spec = util.spec_from_file_location(self.get_import_module(), self._filename)
        module = util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return getattr(module, self._func_name)
    
    def get_import_path(self):
        return os.path.dirname(self._filename)

    def get_import_module(self):
        return self._filename.split("/")[-1].split(".py")[0]


class PythonPicklableFunctionImporter(PythonFunctionImporter):

    def __init__(self, python_function):
        self._python_function = python_function
    @property
    def python_function(self):
        return self._python_function


class PythonLocalFunctionImporter(PythonFunctionImporter):

    _imports_folder_base = "matcal_python_imports"

    def __init__(self, python_function):
        self._func_name = self._get_function_name(python_function)
        import_folder_path = self._create_imports_folder()
        self._filename = os.path.abspath(os.path.join(import_folder_path, self._func_name+".py"))
        self.write_python_function_to_file(python_function)
        self._check_function_can_be_imported()

    @staticmethod
    def _get_function_name(python_function):
        getmembers_list_index = 0
        value_index = 1
        members_list = inspect.getmembers(python_function, inspect.iscode)
        function_name_value_tuple = members_list[getmembers_list_index]
        function_code_object = function_name_value_tuple[value_index]
        if function_code_object.co_name == "<lambda>":
            return inspect.getsource(python_function).split("=")[0].strip()
        else:
            return function_code_object.co_name

    def write_python_function_to_file(self, python_function):
        function_string = inspect.getsource(python_function)
        function_string = PythonLocalFunctionImporter.remove_function_leading_white_space(function_string)
        with open(self._filename, 'w') as source_file:
            source_file.write(function_string)

    staticmethod
    def remove_function_leading_white_space(function_string):
        leading_characters_to_remove = len(function_string.split("\n")[0]) - len(function_string.split("\n")[0].lstrip())
        if leading_characters_to_remove > 0:
            new_string = ""
            for line_number, line in enumerate(function_string.split("\n")):
                new_string += line[leading_characters_to_remove:] +"\n"
            
            return new_string.rstrip("\n")
        else:
            return function_string

    def _create_imports_folder(self):
        if not os.path.exists(self._imports_folder_base):
            os.mkdir(self._imports_folder_base)
        return os.path.abspath(self._imports_folder_base)
