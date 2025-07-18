"""
The residual module contains all classes and functions for residual
calculation. Most are internal tools not intended for users. However, 
the classes related to residual weighting are user facing tools.
"""

import numpy as np
from abc import ABC, abstractmethod
from collections import OrderedDict
import numbers

from matcal.core.data import convert_dictionary_to_data
from matcal.core.python_function_importer import python_function_importer

from matcal.core.logger import initialize_matcal_logger
logger = initialize_matcal_logger(__name__)


class _ResidualCalculatorBase(ABC):

    def __init__(self, *fields_of_interest):
        self._check_fields_of_interest(fields_of_interest)
        self._fields_of_interest = fields_of_interest

    class FieldNameTypeError(RuntimeError):
        """"""

    class FieldsOfInterestNotDefined(RuntimeError):
        """"""

    @abstractmethod
    def calculate(self, reference_data, evaluation_data):
        """"""

    @property
    @abstractmethod
    def fields_of_interest(self):
        """"""

    def _check_item_is_correct_type(self, item, desired_type, message):
        if not isinstance(item, desired_type):
            raise self.FieldNameTypeError(
                "The objective expected a {} type for the added {}, but received object of type {}".format(
                    desired_type, message, type(item)))

    def _check_fields_of_interest(self, fields):
        if len(fields) == 0:
            raise self.FieldsOfInterestNotDefined()
        for field in fields:
            self._check_item_is_correct_type(field, str, "field of interest")

class _ResidualCalculatorBaseWithChecks(_ResidualCalculatorBase):

    class DataMissingFieldsOfInterest(RuntimeError):
        def __init__(self, field_name, data_name):
            message = "{} was not found in the data set {}.".format(field_name, data_name)
            super().__init__(message)

    class InconsistentDataError(RuntimeError):
        pass

    def calculate(self, reference_data, evaluation_data):
        residual_dict = OrderedDict()
        for field in self._fields_of_interest:
            self._check_data_sets_have_field(evaluation_data, field, reference_data)
            ref_field = reference_data[field]
            eval_field = evaluation_data[field]

            n_eval_field = len(np.atleast_1d(eval_field))
            n_ref_field = len(np.atleast_1d(ref_field))
            if n_eval_field != n_ref_field:
                message = f"\nThe two QOI data sets being compared are not the same length  for field \"{field}\"."
                message += f"\n    Reference Length: {n_ref_field}\n    Evaluation Length: {n_eval_field}"
                raise self.InconsistentDataError(message)
            
            residual_dict[field] = self._calculate_field_residual(ref_field, eval_field)
        
        residual = convert_dictionary_to_data(residual_dict)
        residual.set_state(evaluation_data.state)

        return residual

    def _check_data_sets_have_field(self, evaluation_data, field, reference_data):
        if field not in reference_data.field_names:
            raise self.DataMissingFieldsOfInterest(field, reference_data.name)
        if field not in evaluation_data.field_names:
            raise self.DataMissingFieldsOfInterest(field, evaluation_data.name)
        
    @property
    def fields_of_interest(self):
        return self._fields_of_interest


class NominalResidualCalculator(_ResidualCalculatorBaseWithChecks):
    """
    Not intended for users. Calculates residual for two data classes being compared.
    """

    @staticmethod
    def _calculate_field_residual(ref_field, eval_field):
        resid = eval_field - ref_field

        return np.atleast_1d(resid)

class LogResidualCalculator(_ResidualCalculatorBaseWithChecks):
    """
    Not intended for users. Calculates residual for two data classes being compared.
    """

    @staticmethod
    def _calculate_field_residual(ref_field, eval_field):
        resid = np.log(eval_field) - np.log(ref_field)

        return np.atleast_1d(resid)


class _DataNormalizerBase(ABC):
    
    @abstractmethod
    def apply(self, data_array:np.ndarray)-> np.ndarray:
        """"""

class SqrtDataSizeNormalizer(_DataNormalizerBase):
    """
    This class when passed to an objective will divide the residual
    by the square root of the number of experimental data points in the data set. 
    The normalization is intended to reduce
    calibration bias that would favor data sets that contain more data points.

    No additional information needs to be passed to this class.
    """

    def apply(self, residual):
        return residual / np.sqrt(len(np.atleast_1d(residual)))

class InvertedSqrtDataSizeNormalizer(_DataNormalizerBase):
    """
    This class when passed to an objective will multiply the residual
    by the square root of the number of experimental data points in the data set. 
    The normalization is intended to reduce
    calibration bias that would favor data sets that contain more data points.

    No additional information needs to be passed to this class.
    """

    def apply(self, residual):
        return residual * np.sqrt(len(np.atleast_1d(residual)))

class LinearDataSizeNormalizer(_DataNormalizerBase):
    """
    This class when passed to an objective will divide the residual
    by the number of experimental data points in the data set. 
    The normalization is intended to reduce
    calibration bias that would favor data sets that contain more data points.

    No additional information needs to be passed to this class.
    """

    def apply(self, residual):
        return residual / len(np.atleast_1d(residual))


class InvertedLinearDataSizeNormalizer(_DataNormalizerBase):
    """
    This class when passed to an objective will multiply the residual
    by the number of experimental data points in the data set. 
    The normalization is intended to reduce
    calibration bias that would favor data sets that contain more data points.

    No additional information needs to be passed to this class.
    """

    def apply(self, residual):
        return residual * len(np.atleast_1d(residual))


class ResidualWeightingBase(ABC):
    """
    Base class for residual weights not intended for users.
    """
    class TypeError(RuntimeError):
        pass

    @abstractmethod
    def apply(self, reference_data, unconditioned_reference_data, residual_data):
        """"""

    def _check_item_is_correct_type(self, item, desired_type, message):
        if not isinstance(item, desired_type):
            raise self.TypeError(
                "The residual weighting class expected a {} type for the added {}, object of type {"
                "}".format(desired_type, message, type(item)))

    def _convert_to_data_with_state(self, weighted_residual, residual):
        weighted_residual = convert_dictionary_to_data(weighted_residual)
        weighted_residual.set_state(residual.state)
        return weighted_residual


class IdentityWeighting(ResidualWeightingBase):
    """
    This class when passed to an objective will return an unaltered residual. This is the default weighting.
    """

    def apply(self, reference_data, unconditioned_reference_data, residual):
        return residual


class ConstantFactorWeighting(ResidualWeightingBase):
    """
    This class when passed to an objective will multiply the residual by a constant scale factor. The goal of this
    weighting is to apply more or less emphasis to specific objectives during a calibration.
    """
    def __init__(self, scale_factor):
        """
        :param scale_factor: The value to scale the residual vector by.
        :type scale_factor: float
        """
        self._check_item_is_correct_type(scale_factor, numbers.Real, "scale factor")

        self._scale_factor = scale_factor

    def apply(self, reference_data, unconditioned_reference_data, residual):
        weighted_residual = OrderedDict()
        for field in residual.field_names:
            weighted_residual[field] = residual[field]*self._scale_factor
        return self._convert_to_data_with_state(weighted_residual, residual)


class UserFunctionWeighting(ResidualWeightingBase):
    """
    This class when passed to an objective will apply a user defined function to the residual. The goal of this
    weighting is to provide the user with a means to emphasize regions of
    their data.
    """
    def __init__(self, independent_field, target_field, weighting_function):
        """
        :param independent_field: The name of the field to use as an independent field in the user function.
        :type independent_field: str

        :param target_field: The name of the residual field to apply the weighting to.
        :type target_field: str

        :param weighting_function: Predefined function with a signature of (independent_field_data, target_field_data,
            target_field_residual) and returns a NumPy array the same length as the target_field_residual. the results
            of this function will replace the residual value in the evaluation. 
            
            .. note::
                The residual values passed to this
                function are conditioned to be near unity, and thus one should not expect the values to be on the scale as
                the data. The independent field data will be on the scale the user supplied in the data originally.

        :type weighting_function: Callable
        """
        self._target_field = target_field
        self._weighting_function_importer = python_function_importer(weighting_function)
        self._independent_field = independent_field

    def apply(self, reference_data, unconditioned_reference_data, residual):
        weighted_residual = OrderedDict()
        for field in residual.field_names:
            if field == self._target_field:
                weighted_residual[field] = self._apply_weight(unconditioned_reference_data[self._independent_field],
                                                                unconditioned_reference_data[self._target_field],
                                             residual[field])
            else:
                weighted_residual[field] = residual[field]
        return self._convert_to_data_with_state(weighted_residual, residual)

    def _apply_weight(self, independent_field_data, target_field_data, target_residual):
        try:
            logger.debug("Applying user residual weighting...")
            results = self._weighting_function_importer.python_function(independent_field_data, 
                                                                        target_field_data, 
                                                                        target_residual)
            logger.debug("Finished applying user weights.")
        except Exception as e:
            logger.error(f"User residual weighting failed with the following error: { repr(e)}")
            raise e
        return results
        

class NoiseWeightingFromFile(ResidualWeightingBase):
    """
    This class when passed to an objective will divide the residual value by the respective noise field defined in
    the datafile. This noise term will be conditioned and scaled in the same matter at the base field. The scaling
    and conditioning will be handled externally to this tool. The field to be weighted needs to be passed to this class.
    This class will expect another data field  with "_noise"
    appended to the end of the field name.
    """
    def __init__(self, target_field):
        """
        :param target_field: The name of the field to weight.
        """
        self._target_field = target_field
        self._noise_field = self._make_noise_field_name(target_field)

    def apply(self, reference_data, unconditioned_reference_data, residual):
        weighted_residual = OrderedDict()
        for field in residual.field_names:
            if self._is_target(field):
                weighted_residual[field] = self._apply_weighting(residual[field], reference_data)
            else:
                weighted_residual[field] = residual[field]
        return self._convert_to_data_with_state(weighted_residual, residual)

    def _apply_weighting(self, field_residual, reference_data):
        return np.divide(field_residual, reference_data[self._noise_field])

    def _make_noise_field_name(self, target_field):
        return target_field+"_noise"

    def _is_target(self, field_name):
        return field_name == self._target_field
    
class NoiseWeightingConstant(ResidualWeightingBase):
    """
    Apply a constant noise weighting across all data points for a given field. 
    This type of weighting is used to emphasize regions of lower noise and discount 
    those of higher noise. This is useful if there are a large number of data sets, 
    and to reduce the size of a residual, the repeat datasets can be collapsed into 
    their mean values and weighted by their noise. 
    
    The residuals will be divided by the value of the noise. If no noise data exists
    or if the noise value is 0, a 1 should be passed. 
    """
    
    def __init__(self, weights_dict:dict = None, **field_noise_levels):
        """
        Initialize weighting by noise. There are two ways to initialize the data.
        By passing keyword arguments with floats the weights are set for all fields 
        across all experimental states. The second way is by passing a dictionary that
        nests the states and field names, allowing for different weights per state.
        
        :param weights_dict: a dictionary of field names and states that defines the 
            noise weights. The dictionary should be structured [field_name][state_name] = value.
            This type of dictionary can be generated from a 
            :class:`~matcal.core.data.DataCollection` 's report_statistics method.   
        :type weights_dict: dict
        
        :param field_noise_levels: fieldname and noise value pairs. Will apply noise level
            across all experimental states.            
        """
        self._check_inputs(weights_dict, field_noise_levels)
        self._noise_levels, self._state_specific = self._assign_noise_levels(weights_dict,
                                                                             field_noise_levels)

    def apply(self, reference_data, unconditioned_reference_data, residual):
        weighted_residual = OrderedDict()
        current_weights = self._get_weights(residual.state)
        for field_name in residual.field_names:
            if field_name in current_weights.keys():
                weighted_residual[field_name] = np.divide(residual[field_name], 
                                                          current_weights[field_name])
            else:
                weighted_residual[field_name] = residual[field_name]
        return self._convert_to_data_with_state(weighted_residual, residual)

    def _check_inputs(self, weights_dict, field_noise_levels):
        if weights_dict != None and len(field_noise_levels) > 0:
            message = "Only one of weights_dict or field_noise_levels can be specified.\n"
            message += "To specific state sepecific scaling please pass in a two level dictionary.\n"
            message += "    Weights[state_name][field_name] = noise\n"
            message += "If state is to be ignored please pass the terms as keyword arguments or as a"
            message += "dictionary in the from:\n"
            message += "    Weights[field_name] = noise"
            raise RuntimeError(message)
        
    def _assign_noise_levels(self, weights_dict, field_noise):
        by_state = False
        if len(field_noise) > 0:
            return field_noise, by_state
        else:
            for first_level_keys in weights_dict:
                if isinstance(weights_dict[first_level_keys], dict): 
                    by_state = True
                if (not isinstance(weights_dict[first_level_keys], dict)) and by_state:
                    msg = "weights_dict must be consistent, if there is one state specified"
                    msg += ", all states must be specified."
                    raise ValueError(msg) 
            return weights_dict, by_state
    
    def _get_weights(self, state):
        if self._state_specific:
            return self._noise_levels[state.name]
        else:
            return self._noise_levels
    
def get_array(residual):
    data_list = []
    for field in residual.field_names:
            data_list.append(np.atleast_1d(residual[field]))
    if len(data_list) > 0:
        return np.concatenate(data_list)
    else:
        return np.array([])