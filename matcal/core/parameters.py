"""
This module contains all classes related to 
study parameters. 
"""
import abc
from collections import OrderedDict
import numbers
from copy import deepcopy 
import numpy as np 

from matcal.core.logger import initialize_matcal_logger
from matcal.core.utilities import matcal_name_format, CollectionBase

logger = initialize_matcal_logger(__name__)

class MatCalPreprocessorsStruct:
  log = 'log'
  unit = 'unit'

MATCAL_PREPROCESSORS = MatCalPreprocessorsStruct()

class Parameter():
  """
  The MatCal Parameter class is used to create parameters for a MatCal study.
  """

  class distributions():
     """
     Supported parameter distributions in MatCal.
     """
     continuous_design = "continuous_design"
     uniform_uncertain = "uniform_uncertain"

  VALID_DISTRIBUTIONS = [distributions.continuous_design, 
                         distributions.uniform_uncertain]

  def __init__(self,name,lower_bound,upper_bound,current_value=None,
               distribution=distributions.continuous_design, units=None, preprocessing_tags=[]):
    """

    :param name: The name for the parameter. This name (wrapped by curly brackets "{name}") is what
           MatCal will replace in external executable simulation input and material files. This name will 
           be passed into PythonModels as the keyword for a keyword argument.
    :type name: str

    :param lower_bound: This provides the lower bound value for the parameter being created.
    :type lower_bound: float

    :param upper_bound: This provides the upper bound value for the parameter being created.
    :type upper_bound: float

    :param current_value: Some MatCal study algorithms require an initial or current value for a parameter. The user can
          provide it here if required. If one is not provided and it is required for an algorithm, it will be set to the
          average of the lower bound and upper bound.
    :type current_value: float

    :param distribution: This specifies the distribution type for the parameters. The distribution type for a parameter is
          required for certain MatCal studies. By default, the distribution is set to "continuous_design". Valid
          distribution types that are currently supported are defined in the class member "VALID_DISTRIBUTIONS".
    :type distribution: str

    :param units: specify units for the parameter, optional
    :type units: str

    :param preprocessing_tags: [Currently in Development] Pass in tags for 
          MatCal based scaling of a parameter. Scaling parameters to the unit
          or log scale has the potential to improve parameter study performance. 
    :type preprocessing_tags: list

    :raises InvalidDistributionError: if an invalid distribution type is passed to the distribution
            parameter.

    :raises InvalidNameError: The name is not a string.

    :raises InvalidRangeError: The initial value is not within the bounds or the lower bound 
      is greater than the upper bound.
    
    :raises InvalidBoundError: If the bounds are not numbers.

    :raises InvalidCurrentValueError: If the current value is not a number.

    :raises InvalidUnitsError: If the units parameter is not a string.
    """
    self._input_check(name, lower_bound, upper_bound, current_value, distribution, units)
    self._lower_bound = float(lower_bound)
    self._upper_bound = float(upper_bound)
    self._name = name
    self._current_value = self._initialize_current_value(current_value)
    self._units = units
    self._distribution = matcal_name_format(distribution).lower()
    self._preprocessing_tags = self._process_tags(preprocessing_tags)
    
  @property
  def name(self):
    """
    :return: Returns the parameter name.
    :rtype: str
    """
    return self._name
  
  def get_current_value(self):
    """
    :return: Returns the parameter initial value.
    :rtype: float
    """
    return self._current_value

  def get_lower_bound(self):
    """
    :return: Returns the parameter lower bound.
    :rtype: float
    """
    return self._lower_bound
  
  def get_upper_bound(self):
    """
    :return: Returns the parameter upper bound.
    :rtype: float
    """
    return self._upper_bound
  
  def get_name(self):
    """
    :return: Returns the parameter name.
    :rtype: str
    """
    return(self._name)
  
  def get_units(self):
    """
    :return: Returns the units for the parameter
    :rtype: str
    """
    return(self._units)
  
  def get_distribution(self):
    """
    :return: Returns the parameter distribution.
    :rtype: str
    """
    return self._distribution

  def set_current_value(self,value):

    self._current_value = value

  @property
  def preprocessing_tags(self):
    """
    Get the a list of tags used for parameter preprocessing done by MatCal 
    explicitly, and not by an external library. 
    :return: Returns a list of preprocessing tags
    :rtype: list
    """
    return self._preprocessing_tags

  def _add_unit_preprocessing(self):
    self._preprocessing_tags.append(MATCAL_PREPROCESSORS.unit)

  def _add_log_preprocessing(self):
    self._preprocessing_tags.insert(0, MATCAL_PREPROCESSORS.log)


  def _process_tags(self, passed_tags):
    _approved_tags = None
    if isinstance(passed_tags, str):
      _approved_tags =  [passed_tags]
    elif isinstance(passed_tags, list):
      _approved_tags =  passed_tags
    else:
      message = "Invlid format for parameter preprocessing tags."
      message += f"\n passed: P{type(passed_tags)}"
      raise RuntimeError(message)
    return deepcopy(_approved_tags)
  
  class InvalidNameError(RuntimeError):
    pass

  class InvalidBoundError(RuntimeError):
    pass

  class InvalidCurrentValueError(RuntimeError):
    pass

  class InvalidRangeError(RuntimeError):
    pass
  
  class InvalidDistributionError(RuntimeError):
    pass

  class InvalidUnitsError(RuntimeError):
    pass

  def _initialize_current_value(self, value):
    if value is None:
      value = (self._upper_bound + self._lower_bound) / 2.
    else:
      value =float(value)
    return value

  def _input_check(self, name, lower_bound, upper_bound, current_value, distribution, units):
    """
    Checks that the parameters passed to the constructor are valid for the constructor. All parameters passed in are
    the same as those passed to the constructor.
    """
    if not isinstance(name, str):
        raise self.InvalidNameError()
    if not isinstance(lower_bound, numbers.Number):
        raise self.InvalidBoundError()
    if not isinstance(upper_bound, numbers.Number):
        raise self.InvalidBoundError()

    if lower_bound >= upper_bound:
        raise self.InvalidRangeError()

    if current_value is not None:
        if not isinstance(current_value, numbers.Number):
            raise self.InvalidCurrentValueError()

        if current_value < lower_bound or upper_bound < current_value:
            raise self.InvalidRangeError()

    if not isinstance(distribution, str):
        raise self.InvalidDistributionError()
    mc_dist = matcal_name_format(distribution).lower()
    if mc_dist not in self.VALID_DISTRIBUTIONS:
        raise self.InvalidDistributionError("Distribution passed to the parameter must be in {}. Curretly passed {}".format(self.VALID_DISTRIBUTIONS,distribution))
    
    if units is not None and not isinstance(units, str):
        raise self.InvalidUnitsError()
          
    return True

  def __eq__(self, other):
      name_equal = self.name == other.name
      upper_bound_equal = self.get_upper_bound() == other.get_upper_bound()
      lower_bound_equal = self.get_lower_bound() == other.get_lower_bound()
      current_value_equal = self.get_current_value() == other.get_current_value()
      distribution_equal = self.get_distribution() == other.get_distribution()
      units_equal = self.get_units() == other.get_units()
      
      all_equal = name_equal and upper_bound_equal and lower_bound_equal \
         and current_value_equal and distribution_equal and units_equal
      
      return all_equal


class ParameterCollection(CollectionBase):
    """
  A collection of :class:`~matcal.core.parameters.Parameter` objects. This is used to combine multiple parameter
  objects so that they can be passed to a MatCal study. MatCal will use all parameters in the parameter collections
  as the study parameters. Currently, a parameter collection requires that all parameters have the same distribution.
  """

    _collection_type = Parameter

    def __init__(self, name, *parameters):
        """
        :param name: The name of the parameter collection.
        :type name: str

        :param parameters: the parameters to be added to the collection.
        :type parameters: list(class:`~matcal.core.parameters.Parameter`)

        :raises CollectionValueError: If name is an empty string.
        :raises CollectionTypeError: If name is not a string and the parameters to be added to the collection are not of
                the correct type.
        """
        self._param_dist = None
        super().__init__(name, *parameters)

    class DifferentDistributionError(RuntimeError):
        pass

    def _check_param_distribution(self, param):
        if self._param_dist != param.get_distribution():
            raise self.DifferentDistributionError()


    def _set_collection_distribution(self, param):
        if self._param_dist is None:
            self._param_dist = param.get_distribution()      

    def add(self, param):
        """
        This adds a :class:`~matcal.core.parameters.Parameter` to the parameter collection.

        :param param: the parameter object to be added to the collection
        :type param: :class:`~matcal.core.parameters.Parameter`
        """
        self._set_collection_distribution(param)
        self._check_param_distribution(param)
        super().add(param)

    def update_from_results(self, final_parameters:dict):
        """
        Updates the initial value of each parameter in a parameter collection using the results from a study. Can be
        used for workflows where the results of a study are used as the initial point for another study.

        :param final_parameters: the results from a completed MatCal Study
        :type final_parameters: dict
        """
        for param in self._items:
            self._items[param].set_current_value(final_parameters[param])

    def update_parameters(self, **parameter_updates):
        """
        Updates the initial value of parameters in a parameter collection using keyword arguments
        where the keyword is the parameter name and the value are the updated parameter initial value.

        :param parameter_updates: the keyword/value pairs that will be used to update the parameter collection.
        """
        for param, value in parameter_updates.items():
            self._items[param].set_current_value(value)

    def get_distribution(self):
        """
        :return: the distribution of the parameter collection.
        """
        return self._param_dist

    def get_current_value_dict(self):
      """
      :return: returns a dictionary where the keys are the parameter names, and the values are the parameter current value
      :rtype: dict
      """
      initial_values = OrderedDict()
      for param, value in self._items.items():
        initial_values[param] = value.get_current_value()

      return initial_values
 
    def assign_all_to_unit_preprocessing(self):
      for param in self._items.values():
        param._add_unit_preprocessing()
 
class UserDefinedParameterPreprocessor:
    """
    Use a python function as a preprocessor that will modify incoming parameters before they are passed to the model.
    It takes two forms of input:

    #. Pass in a locally defined function for the parameter function.
    #. Pass in the name of the python function for the parameter function as a string and a string which gives
       the full path of the file where the python function is defined. Since MatCal will import from this file,
       it is recommended that nothing is defined or executed in the global names space of that file.

    The python function should take in a dictionary that contains all calibration parameters with keys being the
    parameter name and values being the current value passed in by the study. It should
    return a dictionary with updated parameter values. Note that this can also be used to add new parameter values
    derived from the passed in parameters. Just add a new key-value pair to the dictionary.

    :param function: locally defined function or name of function defined in another file.
    :type function: Callable or str

    :param filename: Name of the file where the function is defined if not in the MatCal python input file.
    :type filename: str

    :return: an updated parameter dictionary with the keys being the parameter names, and the values being the updated
        values. Note that new values can be added, and these values can be strings derived from the incoming parameters.
    :rtype: dict
    """

    def __init__(self, function, filename=None):
        from matcal.core.python_function_importer import python_function_importer
        self._function_importer = python_function_importer(function, filename)
    
    def __call__(self, *args, **kwargs):
        try:
            logger.debug("Preprocessing parameters...")
            results = self._function_importer.python_function(*args, **kwargs)
            logger.debug("Finished preprocessing parameters.")
        except Exception as e:
            logger.error(f"Parameter preprocessor failed with the following error: {repr(e)}")
            raise e
        return results


def serialize_parameter(parameter):
  """
  Converts a parameter into a dictionary that can be serialized 
  with :func:`matcal.core.serializer_wrapper.matcal_save`

  :param parameter: a parameter to be serialized
  :type parameter: :class:`~matcal.core.parameters.Parameter`

  :return: a dictionary with all of the parameter attributes 
  :rtype: dict
  """
  s_p = vars(deepcopy(parameter)) #vars alters passed object
  for key in list(s_p.keys()):
     if key[0] == "_":
        new_key = key[1:]
        s_p[new_key] = s_p.pop(key)
  return s_p

def serialize_parameter_collection(parameter_collection):
    """
    Converts a parameter collection into a dictionary that can be serialized 
    with :func:`matcal.core.serializer_wrapper.matcal_save`

    :param parameter_collection: a parameter collection to be serialized
    :type parameter_collection: :class:`~matcal.core.parameters.ParameterCollection`

    :return: a dictionary with all of the parameter and parameter collection attributes 
    :rtype: dict
    """
    ser_pc = [parameter_collection.name]
    for param in parameter_collection.values():
      ser_pc.append(serialize_parameter(param))
    return ser_pc

def _convert_serialized_parameter(ser_param):
  p = Parameter(**ser_param)
  return p

def _convert_serialized_parameter_collection(ser_param_collection):
    params = []
    name = ser_param_collection[0]
    for ser_param in ser_param_collection[1:]:
      params.append(_convert_serialized_parameter(ser_param))
    return ParameterCollection(name, *params)
  
class UnitParameterScaler:
  
    def __init__(self, param_collection)->None:
      self._to_unit_functions = OrderedDict()
      self._from_unit_functions = OrderedDict()
      self._make_functions(param_collection)
      self._unit_collection = self._make_unit_collection(param_collection)

    @property
    def unit_parameter_collection(self)-> ParameterCollection:
      return self._unit_collection
    
    def to_unit_scale(self, parameter_dict):
      return self._scale_values(parameter_dict,  self._to_unit_functions)
      

    def _scale_values(self, parameter_dict, function_dict):
        out_dict = OrderedDict()
        for p_name, values in parameter_dict.items():
          out_dict[p_name] = function_dict[p_name](values)
        return out_dict
    
    def from_unit_scale(self, parameter_dict):
      return self._scale_values(parameter_dict,  self._from_unit_functions)

    
    def _make_unit_collection(self, param_collection):
      param_collection_name = param_collection.name+"_unit"
      unit_param_collection = ParameterCollection(param_collection_name)
      for p in param_collection:
        guess0 = param_collection[p].get_current_value()
        lower = param_collection[p].get_lower_bound()
        upper = param_collection[p].get_upper_bound()
        unit_values = self._to_unit_functions[p](np.array([lower, upper, guess0]))
        unit_param_collection.add(Parameter(p, *unit_values))
      return unit_param_collection
        
    def _make_functions(self, param_collection):
      for p_name in param_collection:
        to_unit_scale, to_unit_translate = self._get_function_parameters(param_collection, p_name)
        self._to_unit_functions[p_name] = self._TranslateThenScale(to_unit_translate, to_unit_scale)
        self._from_unit_functions[p_name] = self._TranslateThenScale(-to_unit_translate * to_unit_scale, 1/to_unit_scale)

    def _get_function_parameters(self, param_collection, p_name):
      parameter = param_collection[p_name]
      if MATCAL_PREPROCESSORS.unit in parameter.preprocessing_tags:
        lower = parameter.get_lower_bound()
        upper = parameter.get_upper_bound()
        to_unit_scale = 1. / (upper- lower)
        to_unit_translate = -lower
      else:
        to_unit_scale = 1.
        to_unit_translate = 0.
      return to_unit_scale,to_unit_translate
        
    class _TranslateThenScale:
      
      def __init__(self, trans, scale):
        self._trans = trans
        self._scale = scale
        
      def __call__(self, values):
        return (values + self._trans) * self._scale
        

def get_parameters_according_to_precedence(state,
                                           model_constants, 
                                           study_parameters={}):
    combined_params = {}
    combined_params.update(state.params)
    combined_params.update(model_constants)
    combined_params.update(study_parameters)

    return combined_params

def get_parameters_source_according_to_precedence(state,
                                           model_constants, 
                                           study_parameters={}):
    combined_params = get_parameters_according_to_precedence(state, model_constants, 
                                                             study_parameters)
    param_source = {}
    for param in combined_params:
       if param in state.params:
          param_source[param] = "state"
       if param in model_constants:
          param_source[param] = "model constant"
       if param in study_parameters:
          param_source[param] = "study"
    return param_source