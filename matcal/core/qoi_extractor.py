"""
This module contains all classes related to 
data QoI extractors. Most are user facing classes that can 
be added to objectives, however, there a few that 
are not intended for users.
"""
from abc import abstractmethod, ABC
import numpy as np
from types import FunctionType

from matcal.core.data import Data, convert_dictionary_to_data
from matcal.core.logger import initialize_matcal_logger
from matcal.core.utilities import (check_value_is_nonempty_str, 
                                   check_item_is_correct_type)

logger = initialize_matcal_logger(__name__)


class QoIExtractorBase(ABC):
    """
    Base class for quantity of interest (QoI) extractors not intended for users.
    """
    class TypeError(RuntimeError):
        pass

    def __init__(self):
        pass

    @property
    @abstractmethod
    def required_experimental_data_fields(self)->list:
        """"""

    @abstractmethod
    def calculate(self, working_data, reference_data, fields):
        """"""

    def clean_up(self):
        pass


class ReturnPassedDataExtractor(QoIExtractorBase):
    """
    This is the default QoI Extractor. The data just passes through it. 
    If it is used in an objective, the objective
    will attempt to subtract the experimental data from the simulation 
    data as it is read in from the files.
    """
    def __init__(self):
        super().__init__()

    @property
    def required_experimental_data_fields(self):
        return []

    def calculate(self, working_data, reference_data, fields):
        return working_data


class _MulticomponentExtractorWrapperBase(QoIExtractorBase):
    def __init__(self):
        self._extractors = {}

    @property
    def required_experimental_data_fields(self):
        all_required_fields = []
        for extractor in self._extractors.values():
            for current_required_fields in extractor.required_experimental_data_fields:
                all_required_fields.append(current_required_fields)
        return all_required_fields

    def calculate(self, working_data, reference_data, fields):
        lookup_name = self._parse_name(working_data)
        return self._extractors[lookup_name].calculate(working_data, reference_data, fields)

    def add(self, ref_data, extractor):
        lookup_name = self._parse_name(ref_data)
        self._extractors[lookup_name] = extractor

    def clean_up(self):
        for extractor in self._extractors.values():
            extractor.clean_up()


class StateSpecificExtractorWrapper(_MulticomponentExtractorWrapperBase):
    """
    This is a general wrapper class that allows for different instances of a 
    qoi extractor to exist for different data
    states. It has the same interface as a QOI extractor, 
    thus should be almost indistinguishable from a regular qoi extractor
    """
    def _parse_name(self, data):
        state_name = data.state.name
        return state_name


class DataSpecificExtractorWrapper(_MulticomponentExtractorWrapperBase):
    """
    This is a wrapper class for a QOI extractor lookup and execution 
    that looks like a QOI extractor such that it is
    invisible to external users
    """
    
    def _parse_name(self, data):
        state = data.state.name
        name = data.name
        return f"{state}-{name}"


class MaxExtractor(QoIExtractorBase):
    """
    The MaxExtractor QoI Extractor will return the field values of all
    fields at the max value of the analyzed
    field. This can useful when calibrating to the peak load or
    displacement at peak load for a solid mechanics
    calibration or the max temperature or time of the max temperature 
    for a thermal calibration.
    """

    def __init__(self, analyzed_field,  max_index=0):
        """
        :param analyzed_field: the maximum of this field is where all data 
            fields will be extracted and returned.
        :type analyzed_field: int

        :param max_index: If there are multiple maximum values for the
            analyzed field, this index can be used to
            select which value to return.
        :type max_index: int

        :raises TypeError:  If the wrong types are passed into
            the constructor.
        """
        check_value_is_nonempty_str(analyzed_field, "analyzed_field", "MaxExtractor")
        check_item_is_correct_type(max_index, int, "max_index", "MaxExtractor")
        self._max_index = max_index
        self._analyzed_field = analyzed_field
        super().__init__()

    @property
    def required_experimental_data_fields(self) -> list:
        return [self._analyzed_field]

    def calculate(self, working_data, reference_data, fields):
        return self._extract_max_values(working_data)

    def _extract_max_values(self, data):
        field_data = np.atleast_1d(data[self._analyzed_field])
        max_value = np.max(field_data)
        max_index = self._get_max_index(field_data, max_value)
        extracted_data = {}
        for field in data.field_names:
            extracted_data[field] = np.atleast_1d(data[field])[max_index]
        qoi_data = convert_dictionary_to_data(extracted_data)
        qoi_data.set_state(data.state)
        return qoi_data

    def _get_max_index(self, data, max_value):
        tol = 1e-10
        close_enough_range = np.max([tol * np.abs(max_value), tol])
        max_indices = np.argwhere(np.abs(data - max_value) < close_enough_range).flatten()
        if len(max_indices) > 1:
            return max_indices[self._max_index]
        else:
            return max_indices[0]


class InterpolatingExtractor(QoIExtractorBase):
    """
    THe InterpolatingExtractor QoIExtractor will return the field values
    of the working data at the independent
    field values of the reference data. If the InterpolatingExtractor 
    is applied to the simulation data,
    the simulation data will be interpolated onto the experimental 
    data independent field values.

    Note: In order to use the interpolation algorithm, the independent 
    variable for interpolation must be 
    monotonically increasing. As a result, MatCal automatically sorts 
    the data so that the independent variable
    is monotonically increasing in order to conform to this requirement. 
    If this is not desired, create
    a UserDefinedExtractor to meet your needs. 
    """
    def __init__(self, independent_field, left=None, right=None, period=None):
        """
        :param independent_field: The field to be used as the independent 
            variable for interpolation.
        :type independent_field: str
        
        :param left: the left parameter as described in the NumPy interp function. 
        :type left: float

        :param right: the right parameter as described in the NumPy interp function
        :type right: float

        :param period: the period parameter as described in the NumPy interp function
        :type period: float

        :raises TypeError:  If the wrong types are passed into the constructor.
        """
        check_value_is_nonempty_str(independent_field, "independent_field", 
                                    InterpolatingExtractor)
        self._independent_field = independent_field
        self._left = left
        self._right = right
        self._period = period

    @property
    def required_experimental_data_fields(self) -> list:
        return [self._independent_field]

    def calculate(self, working_data, reference_data, fields):
        independent_ref_data = reference_data[self._independent_field]
        independent_working_data = working_data[self._independent_field]
        monotonic_increase_arg_sorting = independent_working_data.argsort()
        sorted_independent_working_data =  independent_working_data[monotonic_increase_arg_sorting]
        interped_data = {self._independent_field: independent_ref_data}
        for field in fields:
            if field is not self._independent_field:
                sorted_dependent_working_data = working_data[field][monotonic_increase_arg_sorting]
                interped_data[field] = _one_dimensional_interpolation(independent_ref_data, sorted_independent_working_data, sorted_dependent_working_data,
                                                                           self._left, self._right, self._period)

        interped_working_data = convert_dictionary_to_data(interped_data)
        interped_working_data.set_state(working_data.state)
        return interped_working_data

def _one_dimensional_interpolation(independent_ref_data, sorted_independent_working_data, sorted_dependent_working_data, left=None, right=None, period=None):
    return np.atleast_1d(np.interp(independent_ref_data,
                                                sorted_independent_working_data,
                                                sorted_dependent_working_data, 
                                                left, right,
                                                period))


class UserDefinedExtractor(QoIExtractorBase):
    """
    The UserDefinedExtractor QoIExtractor will extract the data according
    to a user specified python function.

    The function is passed three arguments: working_data, reference data and 
    the fields of interest for the
    objective. The working data is the data from which
    the QoIs must be extracted.
    The reference data is the data the working data is 
    being compared to. For
    example, if the extractor is applied to the simulation data, 
    the reference data is the corresponding
    experimental data. The working and reference data are passed 
    as :class:`~matcal.core.data.Data` objects and
    the function must return a dictionary with keys corresponding 
    to the fields of interest
    for the objective.
    """

    def __init__(self, function, *required_experiment_fields):
        """
        :param function: a callable function that takes the working 
            :class:`~matcal.core.data.Data` object,
            reference :class:`~matcal.core.data.Data` object, and a 
            list of strings containing the field names the
            extractor must return extracted data for. The function must 
            return data
            as a dictionary with keys for all fields of interest.
            The function is as follows::

                def my_qoi_extractor_function(working_data, reference_data, return_keys_list):
                    working_qois = {}

                    #Do something with reference_data and working_data to calculate working qois.
                    #If needed, verify string in return_keys_list are in working_qois.field_names.

                    return working_qois

        :type function: FunctionType
                    
        :param required_experiment_fields: list of strings that denote data experimental 
            data fields are required for the QoI extractor to perform its QoI extraction.
        :type required_fields: list(str)

        :raises TypeError: if the function is not callable
        """
        check_item_is_correct_type(function, FunctionType, "UserDefinedExtractor",
                                    "function")
        for field in required_experiment_fields:
            check_value_is_nonempty_str(field, "required_experiment_field", 
                                        "UserDefinedExtractor")
        self._function = function
        self._required_fields = required_experiment_fields
        super().__init__()
    
    @property
    def required_experimental_data_fields(self) -> list:
        return self._required_fields

    def calculate(self, working_data, reference_data, fields):
        try:
            extracted_data = self._function(working_data, reference_data, fields)
        except Exception as exc:
            import traceback
            logger.error("Error evaluating user defined QoI extractor.\n")
            logger.error(repr(traceback.format_exception(exc)))
            raise exc
        extracted_data = self._check_and_update_user_function_return_type(extracted_data)
        return  np.atleast_1d(extracted_data)
    
    def _check_and_update_user_function_return_type(self, extracted_data):
        err_str = ("Invalid type returned from the UserDefinedExtractor. It must return a " +
                   "dictionary, numpy structured array, or MatCal Data class, " +
                   f"but a {type(extracted_data)} was returned.")
        if isinstance(extracted_data, dict):
            extracted_data = convert_dictionary_to_data(extracted_data)
        elif isinstance(extracted_data, (np.ndarray, np.record)):
            if not isinstance(extracted_data.dtype, tuple) and not isinstance(extracted_data, Data):
                raise TypeError(err_str)
        else:
            raise TypeError(err_str)
        return extracted_data
