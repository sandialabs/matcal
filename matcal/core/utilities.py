from abc import ABC, abstractmethod
import chardet 
from collections import OrderedDict
import copy
import datetime
import getpass
from numbers import Integral, Real
import numpy as np
import os
from packaging import version

import re
import shutil
from time import sleep
from scipy.interpolate import interp1d


from matcal.core.logger import initialize_matcal_logger
logger = initialize_matcal_logger(__name__)


class MatcalNameFormatError(RuntimeError):
    def __init__(self, text_string):
        message = "Cannot convert to matcal name format, entry must be"
        " a nonzero length string. Current type is {" \
                  "}".format(
            type(text_string))
        super().__init__(message)


def matcal_name_format(to_be_formatted):

    if not isinstance(to_be_formatted, list):
        if not isinstance(to_be_formatted, str) or len(to_be_formatted) < 1:
            raise MatcalNameFormatError(to_be_formatted)
        check_valid_matcal_name_string(to_be_formatted)
        formatted_name = to_be_formatted.replace(' ', '_')
    else:
        if len(to_be_formatted) < 1:
            raise MatcalNameFormatError(to_be_formatted)
        formatted_name = []
        for text_string_item in to_be_formatted:
            formatted_name.append(matcal_name_format(text_string_item))
    return formatted_name


def remove_directory(dir_path):
    removed = False
    attempts = 0
    while not removed and os.path.isdir(dir_path):
        try:
            shutil.rmtree(dir_path)
            removed = True
        except Exception as e:
            sleep(4)
            attempts +=1
            if os.path.isdir(dir_path) and attempts >= 10:
                logger.warning(f"Could not delete directory: {dir_path}... Continuing anyway")
                logger.warning(f"{repr(e)}")
                return


class MatCalTypeStringError(RuntimeError):
    pass


def check_valid_matcal_name_string(string):
    if "/" in string:
        raise MatCalTypeStringError(f"The string \"{string}\" is an invalid MatCal name."
                                    " No backslashes in MatCal name strings.")
    return string


def get_current_files_path(F):
    path = os.path.dirname(F)
    path = os.path.abspath(path)
    return path


def get_username_from_environment():
    return getpass.getuser()


def make_clean_dir(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.mkdir(dir_path)
    return dir_path

def set_significant_figures(x, p):
    x = np.asarray(x)
    x_positive = np.where(np.isfinite(x) & (x != 0), np.abs(x), 10**(p-1))
    mags = 10 ** (p - 1 - np.floor(np.log10(x_positive)))
    return np.round(x * mags) / mags


class CollectionBase(ABC):
    """Base class for all Collections."""

    class CollectionTypeError(RuntimeError):
        """"""

    class CollectionValueError(RuntimeError):
        """"""

    @property
    @abstractmethod
    def _collection_type(self):
        """"""

    def __init__(self, name, *items):
        """
        :raises CollectionValueError: If name is a an empty string.
        :raises CollectionTypeError: If name is not a string and
            the items to be added to the collection are not of
            the correct type.
        """
        self._name = None
        self.set_name(name)
        self._items = OrderedDict()
        self._populate_items(items)

    def _populate_items(self, items):
        for item in items:
            self.add(item)

    def _check_name(self, name):
        if not isinstance(name, str):
            raise self.CollectionTypeError(
                f"Name passed to collection must be an instance of {str}." 
                f" Passed  {name} which is of type  {type(name)}")
        if name == "":
            raise self.CollectionValueError("Name passed to collection must not be empty.")

    def _check_item_is_correct_type(self, item):
        if not isinstance(item, self._collection_type):
            raise self.CollectionTypeError(
                f"Item passed to collection must be an instance of {self._collection_type}. "
                 f"But received object  of type  {type(item)}")

    def add(self, item):
        self._check_item_is_correct_type(item)
        if item.name not in self._items.keys():
            self._items[item.name] = item
        else:
            raise KeyError(f"The item \"{item.name}\" added to Collection \"{self._name}\""
                           " is already in the " 
                         f"collection. Check input.")
            
    def get_item_names(self):
        """
        :return: a list of the names of all items added to the collection.
        """
        if len(self._items) < 1:
            return None
        else:
            return list(self._items.keys())

    def get_number_of_items(self):
        """
        :return: the number of items in the collection
        """
        return len(self._items)

    def dict(self):
        """
        :return: the collection as a dictionary of items with name/value pairs.
        """
        return self._items

    def items(self):
        """
        :return: a list of tuples of key, value pairs contained in the collection.
        """
        return self._items.items()

    def keys(self):
        """
        :return: a list of all available keys in the collection.
        """
        return self._items.keys()

    def set_name(self, name):
        """
        Sets the name of the collection.

        :param name: the new collection name
        :type name: str
        """
        self._check_name(name)
        self._name = name

    def values(self):
        """
        :return: a list of all values in the collection.
        """
        return self._items.values()

    @property
    def name(self):
        """
        :return: the name of the collection
        :rtype: str
        """

        return self._name

    def __len__(self):
        return self._items.__len__()

    @classmethod
    def get_collection_type(self):
        """
        :return: the data type the collection stores
        """
        return self._collection_type

    def __getitem__(self, key):
        return self._items[key]

    def __contains__(self, item):
        return item in self._items

    def __iter__(self):
        return iter(self._items)

    def __add__(self, other):
        new_collection = self.__class__(self.name + " " + other.name)
        for key, item in other.items():
            new_collection.add(item)

        for key, item in self.items():
            new_collection.add(item)

        return new_collection

    def pop(self, key):
        self._items.pop(key)

    def __str__(self):
        keys_list = []
        item_list = []

        for key, item in self.items():
            keys_list.append(key)
            item_list.append(item)

        return "name:{}\n value names: {}\n values: {}\n".format(self.name, keys_list, item_list)

    def __eq__(self, other):
        name_equal = self.name == other.name            
        if len(self._items) != len(other._items):
            return False

        for self_key, self_value in self._items.items():
            if self_key not in other:        
                return False
            if self_value != other[self_key]:
                return False
        return name_equal


class ContainerCollectionBase(CollectionBase):
    """
    A base class for collections that store containers. This is used when
    a key to access an item can or must return
    multiple items.
    """

    def __init__(self, name, *items):
        super().__init__(name, *items)

    def add(self, key, item):
        super()._check_item_is_correct_type(item)
        if key in self._items.keys():
            self._items[key].append(item)
        else:
            self._items[key] = [item]

    def __add__(self, other):
        new_collection = self.__class__(self.name + " " + other.name)
        for key, item in other.items():
            for value in item:
                new_collection.add(value)

        for key, item in self.items():
            for value in item:
                new_collection.add(value)

        return new_collection

    def __str__(self):
        keys_list = []
        item_list = []

        for key, item in self.items():
            keys_list.append(key)
            item_list.append(item)

        return "name:{}\n value names: {}\n values: {}\n".format(self.name, keys_list, item_list)

    def __eq__(self, other):
        names_equal = (self.name == other.name)
        keys_equal = self._check_for_equal_keys(other)
        if not keys_equal:
            return False
        lengths_equal = self._check_for_equal_lengths(other)
        containers_equal = self._check_containers_equal(other)

        all_components_equal = (names_equal and lengths_equal and keys_equal and containers_equal)

        return all_components_equal

    def _check_containers_equal(self, other):
        containers_equal = True
        for key, self_list in self._items.items():
            if len(self_list) != len(other[key]):
                containers_equal = False
                break
            for element in self_list:
                element_found = False
                for element_other in other[key]:
                    
                    if self._check_if_elements_equal(element, element_other):
                        element_found = True
                if not element_found:
                    containers_equal = False
        return containers_equal

    @staticmethod
    def _check_if_elements_equal(element, element_other):
        elements_equal = False
        e_keys = list(element.keys())
        eo_keys = list(element_other.keys())
        if len(e_keys) != len(eo_keys):
            return False
        for key in e_keys:
            if key not in eo_keys:
                return False
        if element.shape:
            if np.array_equal(element, element_other):
                elements_equal = True
        else:
            if element == element_other:
                elements_equal = True
        return elements_equal


    def _check_for_equal_keys(self, other):
        keys_equal = True
        for self_key in self._items.keys():
            if self_key not in other:
                keys_equal = False
        return keys_equal

    def _check_for_equal_lengths(self, other):
        equal_lengths = True
        if len(self._items) != len(other):
            equal_lengths = False
        return equal_lengths
    

def get_current_time_string():
    return str(datetime.datetime.now())


def check_item_is_correct_type(item, desired_type, parent_call_name, 
                               passed_parameter_name, error=TypeError):
    if not isinstance(item, desired_type):
        error_msg =  (f"The parameter \"{passed_parameter_name}\" "+
                      f"passed to \"{parent_call_name}\" " 
                      + f"must be an instance of {desired_type}.\nBut received object " +
                        f" of type {type(item)}.")
        raise error(error_msg)


def check_value_is_positive(value, value_name, parent_call_name):
    if value < 0:
        raise ValueError(f"A positive value must be input for \"{value_name}\" in " +
                         f" \"{parent_call_name}\". Received a " +
                          f"value of {value}.")


def check_value_is_between_values(value, lower_bound, upper_bound, value_name, 
                                  parent_call_name, closed=False):
    if closed:
        not_between = value < lower_bound or value > upper_bound
        msg = (f"A value greater than or equal to {lower_bound} and less than " +
               f"or equal to {upper_bound} must be input for the " +
               f"\"{value_name}\" parameter, but received a value " +
               f"of {value} in function call: \"{parent_call_name}\".")
    else:
        not_between = value <= lower_bound or value >= upper_bound
        msg = (f"A value greater than {lower_bound} and less than "
               f"{upper_bound} must be input for the "
               f"\"{value_name}\" parameter, but received a value "
               f"of {value} in function call:\n\"{parent_call_name}\".")
    if not_between:
        raise ValueError(msg)


def check_value_is_positive_integer(value, value_name, parent_call_name):
    check_item_is_correct_type(value, Integral, parent_call_name, 
                               value_name)
    check_value_is_positive(value, value_name, parent_call_name)


def check_value_is_positive_real(value, value_name, parent_call_name):
    check_item_is_correct_type(value, Real, parent_call_name, 
                               value_name)
    check_value_is_positive(value, value_name, parent_call_name)

def check_value_is_array_like_of_reals(values, values_name, parent_call_name, 
                                       top_level=True):
    valid_array_like = (list, tuple, np.ndarray)
    if isinstance(values, valid_array_like):
        for idx, val in enumerate(values):
            check_value_is_array_like_of_reals(val, values_name+f"[{idx}]", 
                                               parent_call_name, top_level=False)
        if len(values) < 1:
            raise ValueError(f"The \"{values_name}\" argument must have a least "
                             f"one value. The container passed to "
                              f"\"{parent_call_name}\" is empty.")
    elif not top_level and not isinstance(values, valid_array_like):
        check_item_is_correct_type(values, Real, parent_call_name, 
                                    values_name)
    elif top_level and not isinstance(values, valid_array_like):
        raise TypeError(f"\"{parent_call_name}\" expected an array-like "
                         f"object for \"{values_name}\". Recieved objective of "
                         f" type \"{type(values)}\"")
    

def check_value_is_real_between_values(value, lower_bound, upper_bound,
                                        value_name, parent_call_name, 
                                        closed=False):
    check_item_is_correct_type(value, Real, parent_call_name, 
                               value_name)
    check_value_is_between_values(value, lower_bound, upper_bound, 
                                  value_name, parent_call_name, 
                                  closed)
    
    
def check_value_is_nonempty_str(value, value_name, parent_call_name):
    check_item_is_correct_type(value, str, parent_call_name, 
                                value_name)
    if len(value) < 1:
        raise ValueError(f"The argument \"{value_name}\" in "
                         f"\"{parent_call_name}\" must be at least"
                            " of length 1.")

def check_value_is_bool(value, value_name, parent_call_name):
    check_item_is_correct_type(value, bool, parent_call_name, 
                                value_name)
    
    
def _tryint(s):
    try:
        return int(s)
    except:
        return s
    

def _alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [ _tryint(c) for c in re.split('([0-9]+)', s) ]


def _sort_numerically(l):
    """ Sort the given list in the way that humans expect.
    """
    l.sort(key=_alphanum_key)
    return l
    
    
def _time_interpolate(reference_time, working_time, working_data):
    if _is_single_time(working_time):
        return working_data
    else:
        interp_kind = _determine_interp_kind(working_time)
        _, indices = np.unique(working_time, return_index=True)    
        working_data = working_data[indices]
        working_time = working_time[indices]
        interp_time_and_space = interp1d(working_time, working_data, bounds_error=False, 
                            fill_value=working_data[-1], axis=0, kind=interp_kind)(reference_time)
        return interp_time_and_space


def _determine_interp_kind(working_time):
    if len(working_time) < 3:
        interp_kind = 'linear'
    else:
        interp_kind = 'quadratic'
    return interp_kind
    
    
def _is_single_time(working_time):
    return working_time.size < 2


def _find_smallest_rect(n_parameters):
    short_axis_length = int(np.floor(np.power(n_parameters, 1/2)))
    if short_axis_length < 1:
        raise RuntimeError(f"Error finding plot dimension with {n_parameters}")
    long_axis_length = copy.deepcopy(short_axis_length)
    
    def _total_spaces(x, y):
        return x*y
    while _total_spaces(short_axis_length, long_axis_length) < n_parameters:
        long_axis_length += 1
    return short_axis_length,long_axis_length


def _sort_workdirs(input_dirs:list)-> list:
    dir_numbers = _strip_workdir_numbers(input_dirs)
    order = np.argsort(dir_numbers).flatten()
    return list(np.array(input_dirs)[order])


def _strip_workdir_numbers(dir_list: list) -> list:
    numbers = []
    for d in dir_list:
        numbers.append(int(d.split('.')[1]))
    return numbers


def _convert_list_of_files_to_abs_path_list(files_list):
    abs_path_files_list = []
    for filename in files_list:
        abs_path_files_list.append(os.path.abspath(filename))
    return abs_path_files_list


def _get_highest_version_subfolder(folder_path):
    subfolders = [f.path for f in os.scandir(folder_path) if f.is_dir()]
    
    version_pattern = re.compile(r'(\d+\.\d+\.\d+)')
    
    highest_version = None
    highest_version_folder = None
    
    for subfolder in subfolders:
        folder_name = os.path.basename(subfolder)
        match = version_pattern.search(folder_name)
        if match:
            folder_version = version.parse(match.group(1))
            if highest_version is None or folder_version > highest_version:
                highest_version = folder_version
                highest_version_folder = subfolder
    
    return highest_version_folder


def is_text_file(file_path):
    if not os.path.exists(file_path):
        return False
    elif os.path.isdir(file_path):
        return False
    with open(file_path, 'rb') as file:
        raw_data = file.read()
        result = chardet.detect(raw_data)
        encoding = result['encoding']
        if encoding is not None:
            return encoding.lower() in ['ascii', 'utf-8', 'utf-16', 'utf-32']
        return False


def get_string_from_text_file(text_filename):
    with open(text_filename, "r") as f:
        contents = f.read()
    return contents