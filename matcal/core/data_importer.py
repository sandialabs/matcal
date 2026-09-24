"""
The classes and functions in this module are intended
to import data into MatCal from external sources for use
in MatCal studies.
"""

import ast
import glob
import numbers
import os
import sys
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

#: File-size threshold (in bytes) above which a warning is logged on import.
#: Defaults to 10 MB.  Can be overridden at module level for testing or
#: site-specific configuration.
LARGE_FILE_THRESHOLD_BYTES: int = 10 * 1024 * 1024  # 10 MB

_scipy_import_error_msg = ""
try:
    from scipy import io as scipy_io
except ImportError as _scipy_import_error:
    scipy_io = None
    _scipy_import_error_msg = str(_scipy_import_error)

from matcal.core.state import SolitaryState, State
from matcal.core.data import Data, DataCollection, convert_dictionary_to_data
from matcal.core.object_factory import ObjectCreator, SpecificObjectFactory
from matcal.core.serializer_wrapper import matcal_load
from matcal.core.utilities import set_significant_figures

from matcal.core.logger import initialize_matcal_logger

logger = initialize_matcal_logger(__name__)


# This function is named with camelcase to look like a class. We want it to look like a class to keep the
# MatCal UI consistent since users generally only interface with classes. In the future, this can be refactored
# to be a factory class potentially using the __new__ method.
def FileData(
    filename: str,
    state: Optional[State] = None,
    file_type: Optional[str] = None,
    import_strings: bool = False,
    drop_NaNs: bool = False,
    **kwargs,
) -> Data:
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

    :param drop_NaNs: If True, any row containing a NaN or Inf in a numeric
        column is dropped before the data is returned. String columns are
        unaffected.

    :type drop_NaNs: bool

    :param kwargs: additional keyword arguments forwarded to the underlying importer
        (e.g. :class:`~matcal.core.data_importer.CSVDataImporter`). For CSV files
        this includes ``comments`` (a single string, e.g. ``"#"``), ``usecols``,
        ``skip_footer``, ``converters``, ``missing_values``, ``filling_values``,
        and ``delimiter``.

    :return: a populated :class:`~matcal.core.data.Data` object.
    """
    _check_filename_type(filename)
    file_type = _get_file_type(filename, file_type)
    return _import_data(
        filename,
        state=state,
        file_type=file_type,
        import_strings=import_strings,
        drop_NaNs=drop_NaNs,
        **kwargs,
    )


def _import_data(filename, state=None, file_type=None, **kwargs):
    try:
        importer = matcal_probe_data_importer_factory.create(
            file_type, filename, **kwargs
        )
    except KeyError as exc:
        raise KeyError(
            'Data file "{}" of type "{}" is not a supported file type.'
            " MatCal supports the following data types:\n{}".format(
                filename, file_type, list(matcal_probe_data_importer_factory.keys())
            )
        ) from exc

    data = importer.load()
    if state is not None:
        data.set_state(state)
    return data


class DataImporterBase(ABC):

    def __init__(
        self,
        filename: str,
        import_strings: bool = False,
        drop_NaNs: bool = False,
        **kwargs,
    ):
        _check_filename_type(filename)
        self._check_file_exists(filename)
        self._filename = filename
        self._import_strings = import_strings
        self._import_options = self._parse_passed_options(**kwargs)
        self._drop_NaNs = drop_NaNs
        self._warn_if_large_file(filename)

    def _check_file_exists(self, filename: str) -> None:
        if os.path.isdir(filename):
            raise FileNotFoundError(
                f'"{filename}" is a directory, not a file. Check input.'
            )
        if not os.path.isfile(filename):
            raise FileNotFoundError(
                f'The file "{filename}" cannot be found to be imported. Check input.'
            )

    @staticmethod
    def _warn_if_large_file(
        filename: str,
        threshold: Optional[int] = None,
    ) -> None:
        """Log a warning when *filename* exceeds the large-file threshold.

        This helps users notice when they are about to load a very large
        data set into memory, which can be problematic during calibration
        or sensitivity studies that evaluate the model many times.

        :param filename: path to the file being imported.
        :param threshold: optional override for
            :data:`LARGE_FILE_THRESHOLD_BYTES`.
        """
        if threshold is None:
            threshold = LARGE_FILE_THRESHOLD_BYTES
        try:
            file_size = os.path.getsize(filename)
        except OSError:
            return
        if file_size > threshold:
            size_mb = file_size / (1024 * 1024)
            logger.warning(
                'The data file "%s" is %.1f MB which exceeds the %.0f MB '
                "advisory threshold. Loading large data files can "
                "significantly increase memory usage, especially during "
                "calibration or sensitivity studies that iterate over "
                "the model many times. Consider down-sampling, reducing "
                "precision, or using the set_results_storage_options() "
                "method on your study to limit stored history.",
                filename,
                size_mb,
                threshold / (1024 * 1024),
            )

    def _inspect_data_and_clean(self, data):
        self._check_file_not_empty(data)
        self._check_file_data_format(data)
        if self._drop_NaNs:
            data = self._drop_NaNs_from_data(data)
            self._check_file_not_empty(data)

        return data

    def _parse_passed_options(self, **kwargs):  # pylint: disable=unused-argument
        return {}

    def _check_file_not_empty(self, data):
        if not data.size:
            raise ValueError('Empty data file: "{}"'.format(self._filename))

    def _check_file_data_format(self, data):
        for col in data.dtype.names:
            self._check_data_is_interpretable(data[col], col)
            if not self._drop_NaNs:
                self._check_data_is_finite(data[col], col)

    def _check_data_is_interpretable(self, data, column):
        if not self._is_data_interpretable(data):
            raise TypeError(self._get_uninterpretable_data_error_message(data, column))

    def _is_data_interpretable(self, data):
        return self._is_number_subclass(data) or (
            data.dtype.kind in ["U", "S"] and self._import_strings
        )

    def _is_data_entry_interpretable(self, data_value):
        is_numeric = isinstance(data_value, numbers.Number)
        is_numeric_string = False
        if not is_numeric and not self._import_strings and isinstance(data_value, str):
            try:
                float(data_value)
                is_numeric_string = True
            except (ValueError, TypeError):
                is_numeric_string = False
        return is_numeric or is_numeric_string

    def _check_data_is_finite(self, data, column):
        if self._is_number_subclass(data):
            if not np.isfinite(data).all():
                raise ValueError(self._get_nonfinite_data_error_message(data, column))

    def _is_number_subclass(self, data):
        return issubclass(data.dtype.type, numbers.Integral) or issubclass(
            data.dtype.type, numbers.Real
        )

    def _drop_NaNs_from_data(self, data):
        nan_rows = []
        for col in data.dtype.names:
            if self._is_number_subclass(data[col]):
                nan_rows += list(self._get_where_data_not_finite(data[col]))
        nan_rows = sorted(set(nan_rows))
        if nan_rows:
            warning_mssg = (
                "The rows with the following indices were removed on import because "
                'NaNs/INFs were found and "drop_NaNs" was set to "True".\n'
                f"{nan_rows}\nThe data contained in these rows were:\n"
                f"{np.array2string(data[nan_rows])}"
            )
            logger.warning("%s", warning_mssg)
            data = np.delete(data, nan_rows, axis=0)
        return data

    def _get_invalid_data_error_message(self, data, column, bad_data_locs):
        error_msg = 'The file "{}" has data for "{}" that is invalid.\n'.format(
            self._filename, column
        )
        error_msg += "The data has entries:\n"
        error_msg += np.array2string(data[bad_data_locs])
        error_msg += "\nFor row indices:\n"
        error_msg += np.array2string(bad_data_locs) + "\n"
        return error_msg

    def _get_uninterpretable_data_error_message(self, data, column):
        err_str = self._get_invalid_data_error_message(
            data, column, self._get_where_data_not_interpretable(data)
        )
        err_str += (
            "\nData must be a valid type: int or float! "
            "Note: strings importable with 'import_strings' argument only.\n"
        )
        return err_str

    def _get_nonfinite_data_error_message(self, data, column):
        err_message = self._get_invalid_data_error_message(
            data, column, self._get_where_data_not_finite(data)
        )
        err_message += "\nData must be finite!\n"
        return err_message

    def _get_where_data_not_interpretable(self, data):
        are_data_interpretable = np.vectorize(
            self._is_data_entry_interpretable, otypes=[bool]
        )
        return np.where(~are_data_interpretable(data))[0]

    def _get_where_data_not_finite(self, data):
        return np.where(~np.isfinite(data))[0]

    @abstractmethod
    def load(self):
        """Load data from the file and return a Data object."""

    @property
    def filename(self):
        return self._filename

    def __eq__(self, other):
        if not isinstance(other, DataImporterBase):
            return NotImplemented
        return self.filename == other.filename


def _get_file_type(filename, file_type):
    if file_type is None:
        _, ext = os.path.splitext(filename)
        if not ext:
            raise ValueError(
                f'Cannot determine file type from "{filename}": no extension found. '
                "Specify the file_type argument explicitly."
            )
        file_type = ext.lstrip(".")
    _check_file_type_is_string(file_type)
    file_type = file_type.lower()
    return file_type


def _check_file_type_is_string(file_type):
    if not isinstance(file_type, str):
        raise TypeError(
            "The file type passed to a data importer must be a string. Received "
            "variable of type {}".format(type(file_type))
        )


def _check_filename_type(filename):
    if not isinstance(filename, str):
        raise TypeError(
            "The filename passed to a data importer must be a string. Received "
            "variable of type '{}'".format(type(filename))
        )


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

        #. comments — a single string specifying the comment character(s) (e.g. ``"#"``).
           Lists and tuples are not supported.
        #. usecols
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
        state = self._initialize_state(state_dict)
        data = self._inspect_data_and_clean(data)
        return Data(data, state, os.path.abspath(self._filename))

    def _read_data_from_file(self):
        self._check_for_dos()
        nskip, state_dict = self._read_csv_header()
        nskip = self._skip_leading_comments(nskip)
        csv_options = self._create_import_options(nskip)
        data = np.genfromtxt(self._filename, **csv_options)
        return data, state_dict

    def _create_import_options(self, nskip):
        opt = {
            "skip_header": nskip,
            "delimiter": ",",
            "names": True,
            "dtype": None,
            "encoding": None,
            "excludelist": None,
            "autostrip": True,
            "deletechars": "",
            "comments": "#",
        }
        opt.update(self._import_options)
        return opt

    def _parse_passed_options(self, **kwargs):
        options = [
            "comments",
            "usecols",
            "skip_footer",
            "converters",
            "missing_values",
            "filling_values",
            "delimiter",
        ]
        option_dict = {}
        for name in options:
            if name in kwargs:
                option_dict[name] = kwargs[name]
        if "comments" in option_dict and not isinstance(option_dict["comments"], str):
            raise TypeError(
                "comments must be a single string (e.g. comments='#'). "
                f"Received type {type(option_dict['comments']).__name__}."
            )
        return option_dict

    def _initialize_state(self, state_dict):
        state = SolitaryState()
        if state_dict is not None:
            state = self._get_state_from_header_state_dict(state_dict)
            logger.debug(
                'Set state "%s" with state variables %s from %s',
                state.name,
                state_dict,
                self._filename,
            )
        return state

    def _read_csv_header(self):
        nskip = 0
        with open(self._filename, encoding="utf-8") as fh:
            line = fh.readline().strip()
            state_dict = self._get_state_variables_from_file_header(line)
            if state_dict is not None:
                nskip += 1
        return nskip, state_dict

    def _skip_leading_comments(self, nskip):
        if self._does_not_have_comments():
            return nskip
        found_no_comment = False
        with open(self._filename, encoding="utf-8") as fh:
            for _ in range(nskip):
                fh.readline()
            while not found_no_comment:
                line = fh.readline()
                if not line:
                    # EOF reached — no non-comment line found; return as-is
                    break
                line = line.strip()
                if self._line_is_not_comment(line):
                    found_no_comment = True
                else:
                    nskip += 1
        return nskip

    def _line_is_not_comment(self, line):
        if not line:
            return True
        return not line.startswith(self._import_options["comments"])

    def _does_not_have_comments(self):
        does_not_have_comments = "comments" not in self._import_options
        return does_not_have_comments

    def _get_state_variables_from_file_header(self, line):
        state_dict = {}
        if self._has_state_information(line):
            try:
                for name, value in ast.literal_eval(line).items():
                    state_dict[name] = self._process_value(value)
            except (ValueError, SyntaxError) as e:
                raise ValueError(
                    f"A dict-like line was detected on the first line of "
                    f'"{self._filename}" but could not be parsed as a state '
                    f"dictionary.\nLine content: {line!r}\nParse error: {e!r}"
                ) from e
        if len(state_dict) == 0:
            state_dict = None
        return state_dict

    def _has_state_information(self, line):
        has_state_information = "{" in line and "}" in line
        return has_state_information

    def _process_value(self, value):
        if isinstance(value, numbers.Real):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                return value.strip()
        return value

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


def _has_dos_newlines(filename: str, chunk_size: int = 8192) -> bool:
    """Return True if *filename* contains DOS/Windows CRLF (\\r\\n) line endings.

    Reads the file in raw-byte chunks so it is memory-efficient on large
    files and correctly handles a \\r\\n pair that straddles a chunk boundary.
    """
    prev = b""
    with open(filename, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                return False
            data = prev + chunk
            if b"\r\n" in data:
                return True
            prev = chunk[-1:]


def _is_dos(filename: str) -> bool:
    """Return True when *filename* has DOS newlines AND the platform is not Windows.

    DOS line endings are only considered an error on Unix/macOS systems where
    tools like dos2unix are available to correct them.  On Windows the files
    are natively compatible so no error is raised.
    """
    if sys.platform.startswith("win"):
        return False
    return _has_dos_newlines(filename)


def _has_invalid_lines(lines: str) -> bool:
    return len(lines) > 0


def _report_invalid_utc_lines(filename: str) -> str:
    """
    Return lines containing non-ASCII bytes, formatted similarly to the old
    Perl command:

        perl -ne 'print "$. $_" if m/[\\x80-\\xFF]/'

    The output includes the 1-based line number followed by the original line.
    """
    invalid_lines = []

    with open(filename, "rb") as f:
        for line_number, raw_line in enumerate(f, start=1):
            if any(byte >= 0x80 for byte in raw_line):
                line = raw_line.decode("latin-1")
                line = line.replace("\r\n", "\n").replace("\r", "\n")
                invalid_lines.append(f"{line_number} {line}")

    return "".join(invalid_lines)


class DOSFileError(RuntimeError):
    def __init__(self, filename: str):
        message = (
            f"{filename}: is a DOS file. Please convert it to"
            " a Unix-type file with a tool like dos2unix.\n"
        )
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
        Loads the Numpy "npy" data. The file must contain a structured array
        with named numeric columns. Object-dtype arrays (which require pickle)
        are not supported.

        :raises TypeError: If the numpy file does not contain a structured array
            with named columns.

        :return:  A data set object built from the NPY file.
        :rtype: :class:`~matcal.core.data.Data`
        """

        raw = np.load(self._filename, allow_pickle=False)
        if raw.dtype.names is None:
            raise TypeError(
                'The numpy file "{}" has no field names. '
                "MatCal can only load a structured or "
                "record array with named columns.".format(self._filename)
            )
        data = self._inspect_data_and_clean(raw)
        return Data(data, name=os.path.abspath(self._filename))


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
        if scipy_io is None:
            raise ImportError(
                "scipy is required to load .mat files. "
                f"Install it with: pip install scipy\n"
                f"Original error: {_scipy_import_error_msg}"
            )
        data_dictionary = scipy_io.loadmat(filename)
        data_dictionary = self._flatten_dictionary(data_dictionary)
        return data_dictionary

    def _flatten_dictionary(self, data_dictionary):
        flat_dict = {}
        for key, value in data_dictionary.items():
            if self._is_field_data_key(key):
                flat_dict[key] = value.flatten()
        return flat_dict

    def _is_field_data_key(self, key):
        return key[:2] != "__" and key[-2:] != "__"


class BatchDataImporter:
    """
    Class to import multiple data files using a regular expression or a list of filenames.

    Keyword arguments (``**filedata_kwargs``) are forwarded to
    :func:`~matcal.core.data_importer.FileData` for each file.

    States parsed from file headers are reconciled across the batch by rounding numeric
    state parameters to a configurable significant-figure precision (default: 6). States
    whose parameters match at the specified precision are combined into a common state.
    """

    class BatchDataImporterStateError(Exception):
        pass

    def __init__(self, filenames, **filedata_kwargs):
        """
        :param filenames: The names/paths of the files containing the data.
            This can be a list of strs or a single string that is a glob pattern.
        :type filenames: list(str) or str

        :param filedata_kwargs: Keyword arguments forwarded to
            :func:`~matcal.core.data_importer.FileData` for each file (e.g., ``comments``,
            ``delimiter``, ``usecols``, ``import_strings``, ``drop_NaNs``, ``file_type``).
        """
        self._filenames = self._set_batch_filenames(filenames)
        self._datas = []
        self._state_vars = None
        self._batch_cache = None

        # forwarded into FileData(...) calls
        self._filedata_kwargs = dict(filedata_kwargs)

        # precision for reconciling states (significant figures)
        self._state_precision = 6

        # fixed states set by user via set_fixed_state_parameters(...)
        self._fixed_state_params = {}
        self._fixed_state_name = None

        # set during _collect(); declared here to satisfy pylint
        self._has_any_file_state: bool = False
        self._fixed_only_state_name: Optional[str] = None

    def set_state_precision(self, precision: int = 6):
        """
        Set the number of significant figures used when reconciling unique states across the batch.

        Numeric state parameters are rounded to this many significant
        figures when determining whether two states are the same;
        states that match for all parameters
        at this precision are combined.

        :param precision: positive integer significant-figure precision (must be >= 1)
        :type precision: int
        """
        if not isinstance(precision, numbers.Integral):
            raise TypeError("state_precision must be an integer")
        if precision < 1:
            raise ValueError("state_precision must be at least 1")
        self._state_precision = int(precision)
        self._batch_cache = None

    def _any_file_has_state(self) -> bool:
        # true if ANY dataset has non-empty state params
        for d in self._datas:
            if not isinstance(d.state, SolitaryState) and len(d.state.params) > 0:
                return True
        return False

    def set_fixed_state_parameters(self, name: Optional[str] = None, **fixed_states):
        """
        Set fixed/additional state variables applied to every imported dataset during
        reconciliation.

        Values must be either strings or numeric real values.

        Optional naming:
        - If *no* state information is found in any imported file (i.e., all data sets have
          :class:`~matcal.core.state.SolitaryState`), then providing ``name`` will cause the
          fixed-state name to be used as the state name for the batch-created state.
        - If state information *is* found in at least one file, then the provided ``name`` is
          ignored and a warning is logged (the per-file state naming is preserved).

        :param name: Optional name for the fixed-only state when no file states exist.
        :type name: str or None

        :param fixed_states: keyword/value pairs to add to the state (e.g., ``P1=5.0, P2="abc"``)
        :type fixed_states: dict
        """
        if name is not None and not isinstance(name, str):
            raise TypeError("Fixed state name must be a string or None")

        # validate values
        for k, v in fixed_states.items():
            if not isinstance(k, str):
                raise TypeError("Fixed state parameter names must be strings")
            if not isinstance(v, (str, numbers.Real)):
                raise TypeError(
                    f"Fixed state parameter '{k}' must be a string or a real number. "
                    f"Received type {type(v)}"
                )

        self._fixed_state_params = dict(fixed_states)
        self._fixed_state_name = name
        self._batch_cache = None

    def _set_batch_filenames(self, filenames):
        filename_list = []
        if not isinstance(filenames, list) and isinstance(filenames, str):
            filename_list = self._get_filenames_from_pattern(filenames)
        elif isinstance(filenames, list):
            if len(filenames) == 0:
                raise ValueError(
                    "BatchDataImporter requires at least one filename. "
                    "An empty list was passed."
                )
            filename_list = filenames
        else:
            raise TypeError(
                "BatchDataImporter only takes a list of filenames "
                "or a regular expression for "
                f'finding file names. "{filenames}" is not a valid option.'
            )

        for filename in filename_list:
            _check_filename_type(filename)
        return filename_list

    def _get_filenames_from_pattern(self, filenames_list):
        pattern = filenames_list
        filenames_list = sorted(glob.glob(pattern))
        if len(filenames_list) == 0:
            raise FileNotFoundError(
                f'The pattern "{pattern}" passed to '
                "the BatchDataImporter matched no files"
            )
        return filenames_list

    def _get_new_state_with_specified_precision(self, state):
        """
        Return a new State derived from `state` with numeric parameters rounded to the
        configured significant-figure precision.

        Special case: if `state` is SolitaryState and fixed state parameters were provided,
        create a batch fixed-only State (name selection handled separately).
        """
        if isinstance(state, SolitaryState):
            return self._create_state_when_no_file_state()

        return self._create_precision_rounded_state(state)

    def _create_precision_rounded_state(self, state):
        """
        Create a new State whose numeric parameters are rounded to the configured
        significant-figure precision, and whose name is constructed consistently with
        the rounded values.
        """
        precision = self._state_precision
        new_state_name, params = self._round_state_params_and_build_name(
            state, precision
        )
        return State(new_state_name, **params)

    def _round_state_params_and_build_name(self, state, precision: int):
        """
        Round numeric state params to the configured precision and build the canonical
        state name tag.
        """
        new_state_name = ""
        params = {}

        for name, value in state.params.items():
            updated_value = self._round_state_param_value(value, precision)
            new_state_name = self._update_new_state_name(
                precision, name, new_state_name, updated_value
            )
            params[name] = updated_value

        return new_state_name.rstrip("_"), params

    @staticmethod
    def _round_state_param_value(value, precision: int):
        """
        Round numeric values to the configured significant-figure precision.
        Non-numeric values are returned unchanged.
        """
        if isinstance(value, numbers.Real):
            return set_significant_figures(value, precision)
        return value

    @staticmethod
    def _update_new_state_name(
        precision, state_parameter_name, state_name, updated_value
    ):
        fmt = str(precision + 6) + "." + str(precision) + "e"
        if isinstance(updated_value, str):
            fmt = "s"
        state_name += "{0}_{1:{2}}_".format(state_parameter_name, updated_value, fmt)
        return state_name

    def _reconcile_states(self):
        self._state_vars = None
        states = self._get_original_states()
        self._verify_all_data_sets_have_the_same_state_variables(states)
        updated_states_data_collection = DataCollection("reconciled states data")
        self._populate_updated_state_data_collection(updated_states_data_collection)
        return updated_states_data_collection

    def _populate_updated_state_data_collection(self, updated_states_data_collection):
        for data in self._datas:
            original_is_solitary = isinstance(data.state, SolitaryState)
            new_state = self._get_new_state_with_specified_precision(data.state)
            # Apply fixed params to *new* states that came from file state
            # headers (not from SolitaryState, which already incorporates them).
            if (
                self._fixed_state_params
                and not original_is_solitary
                and new_state.name not in updated_states_data_collection.state_names
            ):
                new_state.update(self._fixed_state_params)
                rebuilt_name = ""
                precision = self._state_precision
                for pname, pvalue in new_state.params.items():
                    rounded = self._round_state_param_value(pvalue, precision)
                    rebuilt_name = self._update_new_state_name(
                        precision, pname, rebuilt_name, rounded
                    )
                new_state = State(rebuilt_name.rstrip("_"), **new_state.params)
            if new_state.name in updated_states_data_collection.state_names:
                data.set_state(updated_states_data_collection.states[new_state.name])
            else:
                data.set_state(new_state)
            updated_states_data_collection.add(data)

    def _verify_all_data_sets_have_the_same_state_variables(self, states):
        for data_file, state in states.items():
            current_state_vars = sorted(list(state.params.keys()))
            if self._state_vars is None:
                self._state_vars = current_state_vars
            else:
                if self._state_vars != current_state_vars:
                    raise self.BatchDataImporterStateError(
                        'The file "{}" has the state variables: {} \nExpected the '
                        "following state variables:\n {}. Check input and "
                        "files.".format(data_file, current_state_vars, self._state_vars)
                    )

    def _get_original_states(self):
        states = {}
        for data in self._datas:
            states[data.name] = data.state
        return states

    @property
    def states(self):
        return self.batch.states

    @property
    def filenames(self):
        return self._filenames

    def _collect(self):
        self._datas = []
        for filename in self._filenames:
            d = FileData(filename, **self._filedata_kwargs)
            self._datas.append(d)

        # decide once per batch
        self._has_any_file_state = self._any_file_has_state()
        self._fixed_only_state_name = self._compute_fixed_only_state_name(
            self._has_any_file_state
        )

    def _compute_fixed_only_state_name(self, has_any_file_state: bool) -> str:
        default_name = "batch_fixed_state"
        if self._fixed_state_name is None:
            return default_name

        if has_any_file_state:
            logger.warning(
                "BatchDataImporter fixed state name '%s' was ignored because state "
                "information was found in at least one imported file.",
                self._fixed_state_name,
            )
            return default_name

        return self._fixed_state_name

    def _create_state_when_no_file_state(self):
        if not self._fixed_state_params:
            return SolitaryState()
        return State(self._fixed_only_state_name, **self._fixed_state_params)

    @property
    def batch(self):
        """
        Imports and collects the data into a :class:`~matcal.core.data.DataCollection`.

        If :class:`~matcal.core.state.State` data is included in the files, the appropriate
        states are assigned to each :class:`~matcal.core.data.Data` object along with any fixed
        state parameters specified using
        :meth:`~matcal.core.data_importer.BatchDataImporter.set_fixed_state_parameters`.

        Data with similar states are combined into single states according to the configured
        state precision
        (see :meth:`~matcal.core.data_importer.BatchDataImporter.set_state_precision`).

        The result is cached and reused on subsequent accesses. The cache is
        invalidated when :meth:`set_state_precision` or
        :meth:`set_fixed_state_parameters` is called.
        """
        if self._batch_cache is None:
            self._collect()
            self._batch_cache = self._reconcile_states()
        return self._batch_cache


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


matcal_probe_data_importer_factory = ProbeDataImporterFactory()
matcal_probe_data_importer_factory.register_creator("csv", CSVProbeImporterCreator())
matcal_probe_data_importer_factory.register_creator("npy", NumpyProbeImporterCreator())
matcal_probe_data_importer_factory.register_creator("mat", MatlabProbeImporterCreator())
matcal_probe_data_importer_factory.register_creator("json", JSONProbeImporterCreator())
