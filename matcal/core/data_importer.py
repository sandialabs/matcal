"""
The classes and functions in this module are intended 
to import data into MatCal from external sources for use 
in MatCal studies.
"""

import os
from matcal.core.object_factory import ObjectCreator, SpecificObjectFactory
from matcal.core.serializer_wrapper import matcal_load
import numpy as np
import subprocess
import glob
from abc import ABC, abstractmethod
import numbers

from matcal.core.state import SolitaryState, State
from matcal.core.data import Data, DataCollection, convert_dictionary_to_data
from matcal.core.utilities import set_significant_figures

from matcal.core.logger import initialize_matcal_logger
logger = initialize_matcal_logger(__name__)


# This function is named with camelcase to look like a class. We want it to look like a class to keep the
# MatCal UI consistent since users generally only interface with classes. In the future, this can be refactored
# to be a factory class potentially using the __new__ method.
def FileData(filename:str, state:State=None, file_type:str=None, 
             import_strings:bool=False, drop_NaNs:bool=False, 
             *args, **kwargs) -> Data:
    """
    A function used to import a MatCal :class:`~matcal.core.data.Data` object 
    from a file. The user needs to use
    this function to load experimental data from a file into MatCal

    :param filename: the name of the file to be loaded.
    :type filename: str

    :param state: optional state to be assigned to the data being imported
    :type state: :class:`~matcal.core.state.State`

    :param file_type: optional file type passed by the user. MatCal will attempt
        to guess the file type based on the
        file extension. MatCal recognizes "csv", "npy" and "mat" file types 
        and only accepts these strings as input for 
        this parameter.
    :type file_type: str

    :param import_strings: A boolean to allow MatCal to read in string data 
        fields. By default it is set to False and
        will error out if any data cannot be converted to numeric values.
    :type import_strings: bool

    :param drop_NaNs: A boolean to allow MatCal to read in string data fields 
        with NaNs by dropping any rows that contain
        a NaN.

    :type import_strings: bool

    :return: a populated :class:`~matcal.core.data.Data` object.
    """
    _check_filename_type(filename)
    file_type = _get_file_type(filename, file_type)
    return _import_data(filename, state=state, file_type=file_type, 
                        import_strings=import_strings, drop_NaNs=drop_NaNs, 
                        *args, **kwargs)


def _import_data(filename, state=None, file_type=None, *args, **kwargs):
    try:
        importer = MatCalProbeDataImporterFactory.create(file_type, filename, 
                                                         *args, **kwargs)
    except KeyError:
        raise KeyError("Data file \"{}\" of type \"{}\" is not a supported file type." \
                       " MatCal supports the following data types:\n{}".format(filename, file_type,
                                                                          list(MatCalProbeDataImporterFactory.keys())))

    data = importer.load()
    if state is not None:
        data.set_state(state)
    return data


class DataImporterBase(ABC):

    def __init__(self, filename: str, import_strings:bool=False, 
                 drop_NaNs:bool=False, **kwargs):
        _check_filename_type(filename)
        self._check_file_exists(filename)
        self._filename = filename
        self._import_strings = import_strings
        self._import_options = self._parse_passed_options(**kwargs)
        self._drop_NaNs = drop_NaNs
        self._rows_with_NaNs = []

    def _check_file_exists(self, filename):
        if not os.path.isfile(filename):
            raise FileNotFoundError(
                f'The file \"{filename}\" cannot be found to be imported. Check input.')

    def _inspect_data_and_clean(self, data):
        self._check_file_not_empty(data)
        self._check_file_data_format(data)
        if self._drop_NaNs:
            data = self._drop_NaNs_from_data(data)

        return data

    def _parse_passed_options(self, **kwargs):
        return {}

    def _check_file_not_empty(self, data):
        if not data.size:
            raise ValueError("Empty data file: \"{}\"".format(self._filename))

    def _check_file_data_format(self, data):
        for col in data.dtype.names:
            self._check_data_is_interpretable(data[col], col)
            if not self._drop_NaNs:
                self._check_data_is_finite(data[col], col)
               
    def _check_data_is_interpretable(self, data, column):
        if not self._is_data_interpretable(data):
            raise TypeError(self._get_uninterpretable_data_error_message(data, column))

    def _is_data_interpretable(self, data):
        return (np.issubdtype(data.dtype, np.integer) or np.issubdtype(data.dtype, np.floating) or \
                (data.dtype.kind in ["U", "S"] and self._import_strings))

    def _is_data_entry_interpretable(self, data_value):
        is_numeric = isinstance(data_value, numbers.Number)
        is_numeric_string = False
        if not is_numeric and not self._import_strings and isinstance(data_value, str):
            is_numeric_string = np.char.isnumeric(data_value) or data_value.lower() in ["inf", "nan"]
        return is_numeric or is_numeric_string

    def _check_data_is_finite(self, data, column):
        if np.issubdtype(data.dtype, np.integer) or np.issubdtype(data.dtype, np.floating):
            if not (np.isfinite(data).all() == True):
                raise TypeError(self._get_nonfinte_data_error_message(data, column))

    def _drop_NaNs_from_data(self, data):
        NaN_rows_to_drop = []
        for col in data.dtype.names:
            NaN_rows_to_drop += list(self._get_where_data_not_finite(data[col])[0])
        NaN_rows_to_drop = list(set(NaN_rows_to_drop))
        warning_mssg = "The rows with the following indices were removed on import because NaNs/INFs were found and \"drop_NaNs\" was set to \"True\".\n"
        warning_mssg += f"{NaN_rows_to_drop}"
        warning_mssg += "\nThe data contained in these rows were:\n"
        warning_mssg += np.array2string(data[NaN_rows_to_drop])
        if NaN_rows_to_drop:
            data = np.delete(data, NaN_rows_to_drop, axis=0)
            logger.warning(warning_mssg)
        return data

    def _get_invalid_data_error_message(self, data, column, bad_data_locs):
        error_msg = 'The file \"{}\" has data for \"{}\" that is invalid.\n'.format(
                        self._filename, column)
        error_msg += "The data has entries:\n"
        error_msg += np.array2string(data[bad_data_locs])
        error_msg += "\nFor row indices:\n"
        error_msg += np.array2string(bad_data_locs) + "\n"
        return error_msg

    def _get_uninterpretable_data_error_message(self, data, column):
        err_str = self._get_invalid_data_error_message(data, column,
                                                       self._get_where_data_not_interpretable(data))
        err_str += ('\nData must be a valid type: int or float! '
                    'Note: strings importable with \'import_strings\' argument only.\n')
        return err_str

    def _get_nonfinte_data_error_message(self, data, column):
        err_message = self._get_invalid_data_error_message(data, column, 
                                                           self._get_where_data_not_finite(data))
        err_message += "\nData must be finite!\n"
        return err_message

    def _get_where_data_not_interpretable(self, data):
        are_data_interpretable = np.vectorize(self._is_data_entry_interpretable, otypes=[bool])
        return np.atleast_1d(np.where(~are_data_interpretable(data)))

    def _get_where_data_not_finite(self, data):
        return np.atleast_1d(np.where(~np.isfinite(data)))

    @abstractmethod
    def load(self, **opts):
        """"""

    @property
    def filename(self):
        return self._filename

    def __eq__(self, other):
        equal_filename = self.filename == other.filename

        return equal_filename


def _get_file_type(filename, file_type):
    if file_type is None:
        file_type = filename.split(".")[-1]
    _check_file_type_is_string(file_type)
    file_type = file_type.lower()

    return file_type


def _check_file_type_is_string(file_type):
    try:
        assert isinstance(file_type, str)
    except AssertionError:
        raise TypeError("The file type passed to a data importer must be a string. Received "
                                      "variable of type {}".format(type(file_type)))


def _check_filename_type(filename):
    try:
        assert isinstance(filename, str)
    except AssertionError:
        raise TypeError("The filename passed to a data importer must be a string. Received "
                                      "variable of type '{}'".format(type(filename)))


class CSVDataImporter(DataImporterBase):
    """
    Class for reading in data from a CSV file. This uses the NumPy "genfromtxt" 
    function to read data in from CSV
    files. It assumes that the columns have headers so that MatCal can identify 
    what information is being read in and
    make appropriate comparisons between simulations and experiments. This is wrapped by
    :func:`~matcal.core.data_importer.FileData`.

    .. note::
        This accepts the following keyword arguments that are valid in Numpy "genfromtxt":

        #. comments
        #. uscols
        #. skip_footer
        #. converters
        #. missing_values
        #. filling_values
    """

    def load(self):
        """
        Loads the CSV data.

        :return:  A data set object built from the CSV file.
        :rtype: :class:`~matcal.core.data.Data`
        """
        data, state_dict = self._read_data_from_file()
        state = self._intiailize_state(state_dict)
        data = self._inspect_data_and_clean(data)
        return Data(data, state, os.path.abspath(self._filename))

    def _read_data_from_file(self):
        self._check_for_dos()
        try:
            nskip, state_dict = self.read_csv_header()
            nskip = self._skip_leading_comments(nskip)
            csv_options = self._create_import_options(nskip)
            data = np.genfromtxt(self._filename, **csv_options)
        except Exception as err:
            raise err("Error occurred while reading data file {}\n {}".format(self._filename, err))       
        return data, state_dict

    def _create_import_options(self, nskip):
        opt = {'skip_header': nskip, 'delimiter': ",", 'names': True, 'dtype': None,
                'encoding': None, 'excludelist': None, 'autostrip': True, 'deletechars': "",
                'comments':"#"}
        opt.update(self._import_options)
        return opt

    def _parse_passed_options(self, **kwargs):
        options = ['comments', 'usecols', 'skip_footer', 'converters', 
                   'missing_values', 'filling_values', "delimiter"]
        option_dict = {}
        for name in options:
            if name in kwargs.keys():
                option_dict[name] = kwargs[name]
        return option_dict

    def _intiailize_state(self, state_dict):
        state = SolitaryState()
        if state_dict is not None:
            state = self._get_state_from_header_state_dict(state_dict)
            logger.debug("Set state \"{0}\" with state variables {1} from {2}".format(state.name, state_dict,
                                                                                      self._filename))
        return state

    def read_csv_header(self):
        nskip = 0
        with open(self._filename) as fh:
            line = fh.readline().strip()
            state_dict = self._get_state_variables_from_file_header(line)
            if state_dict is not None:
                line = fh.readline().strip()
                nskip += 1
        return nskip, state_dict

    def _skip_leading_comments(self, nskip):
        if self._does_not_have_comments():
            return nskip
        found_not_comment = False
        with open(self._filename) as fh:
            for i in range(nskip):
                line = fh.readline().strip()
            while not found_not_comment:
                line = fh.readline().strip()
                if self._line_is_not_comment(line):
                    found_not_comment = True
                else:
                    nskip += 1
        return nskip

    def _line_is_not_comment(self, line):
        line_is_not_comment = line[0] != self._import_options['comments']
        return line_is_not_comment

    def _does_not_have_comments(self):
        does_not_have_comments = 'comments' not in self._import_options.keys()
        return does_not_have_comments

    def _get_state_variables_from_file_header(self, line):
        state_dict = {}
        if self._has_state_information(line):
            for name, value in eval(line).items():
                state_dict[name] = self._process_value(value)
        if len(state_dict) == 0:
            state_dict = None
        return state_dict

    def _has_state_information(self, line):
        has_state_information = "{" in line and "}" in line and ":" in line
        return has_state_information

    def _process_value(self, value):
        if isinstance(value, (float, int, np.double)):
            return float(value)
        elif isinstance(value, str) and value.isnumeric():
            return float(value)
        else:
            return value.strip()

    def _get_state_from_header_state_dict(self, state_dict):
        state_name = self._create_state_name_from_state_dict(state_dict)
        state = State(state_name, **state_dict)
        return state

    def _create_state_name_from_state_dict(self, sdict):
        names = list(sdict.keys())
        names.sort()
        tag = ""
        for name in names:
            tag += self._format_item(name, sdict)
        tag = tag.rstrip("_")
        return tag

    def _format_item(self, name, sdict):
        formatted_item = f"{name}_"
        v = sdict[name]
        if isinstance(v, str):
            formatted_item += f"{v}_"
        else:
            formatted_item += "{0:12.6e}_".format(float(v))
        return formatted_item
    
    def _check_for_dos(self):
        if _is_dos(self.filename):
            raise DOSFileError(self.filename)
        invalid_lines = _report_invalid_utc_lines(self.filename)
        if _has_invalid_lines(invalid_lines):
            raise InvalidCharacterError(self.filename, invalid_lines)
        

def _is_dos(filename: str)-> bool:
    unix_report_as_dos = _unix_detect_dos(filename)
    mac_report_as_dos = _mac_detect_dos(filename)
    return unix_report_as_dos or mac_report_as_dos

def _mac_detect_dos(filename:str)->bool:
    dos_newline = "\\r\\n'"
    with open(filename, 'r', newline=None) as f:
        #loop to avoid potential header character issues
        for i in range(2):
            try:
                line = f.readline()
            except Exception:
                pass
        
        
        new_lines = repr(f.newlines)
        has_dos = dos_newline in new_lines
    return has_dos


def _has_invalid_lines(lines: str)->bool:
    return len(lines) > 0


def _unix_detect_dos(filename: str) -> bool:
    unix_report = _get_unix_file_report(filename)
    return "CRLF" in unix_report.split()


def _get_unix_file_report(filename):
    file_commands = ["file", filename]
    command_result = subprocess.run(file_commands, stdout=subprocess.PIPE)
    unix_report = command_result.stdout.decode('utf-8').split(":")[1].strip()
    return unix_report


def _report_invalid_utc_lines(filename: str) ->str:
    print_string = 'print "$. $_" if m/[\x80-\xFF]/'
    commands = ["perl", "-ne", f"{print_string}" , filename]
    command_result = subprocess.run(commands, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    b_string = command_result.stdout
    return b_string.decode('unicode_escape')


class DOSFileError(RuntimeError):
    def __init__(self, filename: str):
        message = f"{filename}: is a DOS file. Please convert it to a unix type file with a tool like dos2unix\n"
        super().__init__(message)


class InvalidCharacterError(RuntimeError):
    def __init__(self, filename: str, lines: str):
        message = f"\n{filename}\nhas unsupported characters in the following locations:\n{lines}"
        super().__init__(message)


class NumpyDataImporter(DataImporterBase):
    """
       Class for reading in data from a numpy ".npy" file. 
       This uses the np.load() function to read data in
       from numpy files. Since field names are required for MatCal, 
       it assumes that a structured array or record is
       saved in the file. If there are no field names, 
       it will fail to load the file. This is wrapped by
       :func:`~matcal.core.data_importer.FileData`.
       """

    def load(self):
        """
        Loads the Numpy "npy" data.

        :raises FileNotFoundError: If the data file is not found.
 
        :return:  A data set object built from the NPY file.
        :rtype: :class:`~matcal.core.data.Data`
        """

        data = np.load(self._filename)  # NOTE assumes pickled dict
        data = self._inspect_data_and_clean(data)
        data = Data(data, name=os.path.abspath(self._filename))
        if data.field_names is None:
            raise TypeError("The numpy file \"{}\" has no field names. "
                            "MatCal can only load a structured or "
                            "record array with named columns.".format(self._filename))

        return data


class MatlabDataImporter(DataImporterBase):
    """
       Class for reading in data from a Matlab ".mat" file. This uses the 
       scipy.io.loadmat() function to read data in
       from the files. Since field names are required for MatCal, it assumes 
       that the data are stored in a format of
       1d vectors with each variable name being the field name. 
       This is wrapped by :func:`~matcal.core.data_importer.FileData`.
       """

    def load(self):
        data_dictionary = self._create_flattened_data_from_mat_file(self._filename)
        data = convert_dictionary_to_data(data_dictionary)
        data = self._inspect_data_and_clean(data)
        data.set_name(os.path.abspath(self._filename))
        return data

    def _create_flattened_data_from_mat_file(self, filename):
        from scipy import io
        data_dictionary = io.loadmat(filename)
        data_dictionary = self._flatten_dictionary(data_dictionary)
        return data_dictionary

    def _flatten_dictionary(self, data_dictionary):
        flat_dict = {}
        for key, value in data_dictionary.items():
            if self._is_field_data_key(key):
                flat_dict[key] = value.flatten()
        return flat_dict

    def _is_field_data_key(self, key):
        front = key[:2]
        back = key[-2:]
        format_string = "__"
        if front != format_string and back != format_string:
            return True
        else:
            return False


class BatchDataImporter:
    """
    Class to import multiple data files using a regular expression or a list of filenames.
    """
    class BatchDataImporterStateError(Exception):
        def __init__(self, *args):
            super().__init__(*args)

    def __init__(self, filenames,  file_type=None, fixed_states=None):
        """
        :param filenames: The names/paths of the file containing the data.
            This can be a list of strs or a single string that is a glob pattern.
        :type filename: list(str) or str

        :param file_type: the file type to be read in. Default is to read the extension. 
             MatCal recognizes "csv", "npy" and "mat" file types 
            and only accepts these strings as input for 
            this parameter.
        :type file_type: str

        :param fixed_states: additional state variables not defined in the data files
        :type fixed: dict
        """
        self._filenames = self._set_batch_filenames(filenames)
        self._datas = []
        self._load_opts = {}
        self._additional_states = {}
        self._state_vars = None
        if fixed_states is not None:
          if not isinstance(fixed_states,dict):
            raise TypeError("The fixed states passed to the BatchDataImporter"
                            " must be in dict format")
          self._additional_states = fixed_states

        self._file_type=None
        if file_type is not None:
          if not isinstance(file_type,str):
            raise TypeError("File type passed to the BatchDataImporter "
                            "must be specified as a string")
          self._file_type = file_type

        self._load_opts = {}
        self._set_default_loading_options()

    def _set_batch_filenames(self, filenames):
        filename_list = []
        if not isinstance(filenames, list) and isinstance(filenames, str):
            filename_list = self._get_filenames_from_pattern(filenames)
        elif isinstance(filenames, list):
            filename_list = filenames
        else:
            raise TypeError("BatchDataImporter only takes a list of filenames or a regular expression for "
                                 "finding file names. \"{}\" is not a valid option.".format(filenames))

        for filename in filename_list:
            self._check_filename_type(filename)
        return filename_list


    def _get_filenames_from_pattern(self, filenames_list):
        pattern = filenames_list
        filenames_list = sorted(glob.glob(pattern))
        if len(filenames_list) == 0:
            raise FileNotFoundError(f"The pattern \"{pattern}\" passed to "
                                    "the BatchDataImporter matched no files")
        return filenames_list

    def set_options(self, **opts):
        """
        This can be used to set options available for batch loading files.
        Currently only "state_precision" is supported and it controls the precision of 
        the state values in reconciling unique states. It has a default value of six.
        As a result, states are kept independent if their precision differs up 
        to the sixth significant
        figure for each state parameter, and states where all parameters have the same values 
        up to the sixth significant figure are combined into a common 
        state as repeats. 

        :param opts: comma delimited list of valid keyword/value pairs.
        """
        self._load_opts.update(opts)

    def _check_filename_type(self, filename):
        if not isinstance(filename, str):
            raise TypeError("The filename passed to the BatchDataImporter must be a string."
            f" Received variable of type '{type(filename)}'")

    def _set_default_loading_options(self):
      self._load_opts["state_precision"] = 6

    def _get_new_state_with_specified_precision(self, state):
      precision = self._load_opts["state_precision"]
      new_state_name = ""
      if isinstance(state, SolitaryState) and self._additional_states:
          new_state_name = "batch_fixed_state"
      elif isinstance(state, SolitaryState):
          return SolitaryState()

      params = {}
      for name, value in state.params.items():
        if isinstance(value, numbers.Real):
            updated_value = set_significant_figures(value, precision+1)
        else:
            updated_value = value
        new_state_name = self._update_new_state_name(precision, name, new_state_name, updated_value)
        params[name] = updated_value
      new_state_name = new_state_name.rstrip("_")
      return State(new_state_name, **params)

    @staticmethod
    def _update_new_state_name(precision, state_parameter_name, state_name, updated_value):

        if precision == 0:
            fmt = str(precision + 6) + "." + str(precision+1) + "e"
        else:
            fmt = str(precision + 6) + "." + str(precision) + "e"
        if isinstance(updated_value, str):
            fmt = "s"
        state_name += "{0}_{1:{2}}_".format(state_parameter_name, updated_value, fmt)
        return state_name

    def _reconcile_states(self):
      states = self._get_original_states()
      self._verify_all_data_sets_have_the_same_state_variables(states)
      updated_states_data_collection = DataCollection("reconciled states data")
      self._populate_updated_state_data_collection(updated_states_data_collection)

      return updated_states_data_collection

    def _populate_updated_state_data_collection(self, updated_states_data_collection):
        for data in self._datas:
            new_state = self._get_new_state_with_specified_precision(data.state)
            if new_state.name in updated_states_data_collection.state_names:
                data.set_state(updated_states_data_collection.states[new_state.name])
            else:
                new_state.update(self._additional_states)
                data.set_state(new_state)
            updated_states_data_collection.add(data)

    def _verify_all_data_sets_have_the_same_state_variables(self, states):
        for data_file, state in states.items():
            current_state_vars = sorted(list(state.params.keys()))
            if self._state_vars is None:
                self._state_vars = current_state_vars
            else:
                if self._state_vars != current_state_vars:
                    raise self.BatchDataImporterStateError("The file \"{}\" has the state variables: {} \nExpected the "
                                                           "following state varaibles:\n {}. Check input and "
                                                           "files.".format(data_file, current_state_vars,
                                                                           self._state_vars))

    def _get_original_states(self):
        states = {}
        for data in self._datas:
            states[data.name] = data.state
        return states

    @property
    def states(self):
      return self._states

    @property
    def filenames(self):
      return self._filenames

    def _collect(self):
      for filename in self._filenames:
        d = FileData(filename, file_type=self._file_type)
        self._datas.append(d)

    @property
    def batch(self):
      """
      Imports and collects the data into a :class:`~matcal.core.data.DataCollection`. 
      If :class:`~matcal.core.state.State` data is included 
      in the files, the appropriate states are assigned 
      to each :class:`~matcal.core.data.Data` class along with any fixed 
      state parameters specified in the 
      :class:`~matcal.core.data_importer.BatchDataImporter` constructor.
      Data with similar states are combined into single states according to
      the "state_precision" optional parameter with a default value of six.
      See :meth:`~matcal.core.data_importer.BatchDataImporter.set_options` for 
      more details.
      """
      self._collect()
      data_collection = self._reconcile_states()
      return data_collection


class JSONProbeDataImporter(DataImporterBase):

    def load(self):
        data_dictionary = matcal_load(self._filename)
        data = convert_dictionary_to_data(data_dictionary)
        data = self._inspect_data_and_clean(data)
        data.set_name(os.path.abspath(self._filename))
        return data


class ProbeDataImporterFactory(SpecificObjectFactory):
    pass

class CSVProbeImporterCreator(ObjectCreator):

    def __call__(self, *args, **kwargs):
        return CSVDataImporter(*args, **kwargs)
    
class NumpyProbeImporterCreator(ObjectCreator):

    def __call__(self, *args, **kwargs):
        return NumpyDataImporter(*args, **kwargs)
    
class MatlabProbeImporterCreator(ObjectCreator):

    def __call__(self, *args, **kwargs):
        return MatlabDataImporter(*args, **kwargs)
    
class JSONProbeImporterCreator(ObjectCreator):

    def __call__(self, *args, **kwargs):
        return JSONProbeDataImporter(*args, **kwargs)


MatCalProbeDataImporterFactory = ProbeDataImporterFactory()
MatCalProbeDataImporterFactory.register_creator('csv', CSVProbeImporterCreator())
MatCalProbeDataImporterFactory.register_creator('npy', NumpyProbeImporterCreator())
MatCalProbeDataImporterFactory.register_creator('mat', MatlabProbeImporterCreator())
MatCalProbeDataImporterFactory.register_creator('json', JSONProbeImporterCreator())