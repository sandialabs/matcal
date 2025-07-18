"""
The objective module contains the classes related to objectives. 
This includes the metric functions, the base objectives, specialized 
objectives, objective sets. User facing classes only include metric functions
and objectives.
"""

from abc import abstractmethod, ABC
from collections import OrderedDict
from copy import deepcopy
from itertools import count

import numbers
import numpy as np


from matcal.core.objective_results import ObjectiveQOI, ObjectiveResults
from matcal.core.qoi_extractor import (InterpolatingExtractor, QoIExtractorBase, 
                                       ReturnPassedDataExtractor) 
from matcal.core.data import (convert_dictionary_to_data, DataCollection,
    RangeDataConditioner)
from matcal.core.residuals import (IdentityWeighting, LogResidualCalculator, 
                                   ResidualWeightingBase, SqrtDataSizeNormalizer, 
                                   LinearDataSizeNormalizer, NominalResidualCalculator, 
                                   InvertedSqrtDataSizeNormalizer, InvertedLinearDataSizeNormalizer)
from matcal.core.state import StateCollection
from matcal.core.utilities import CollectionBase, check_value_is_array_like_of_reals

from matcal.core.logger import initialize_matcal_logger
logger = initialize_matcal_logger(__name__)


class _MetricFunctionBase(ABC):
    """
    Base class for all metric functions. Metric functions convert the residual 
    vector into a single scalar objective function value.
    """

    def __init__(self):
        """"""""

    def normalize_and_calculate(self, residuals):
        normalized_residual = self._normalize(residuals)
        return self._calculate(normalized_residual)

    def normalize_only(self, residual):
        return self._normalize(residual)

    def calculate_only(self, residual):
        return self._calculate(residual)

    @abstractmethod
    def calculate_group_normalization_factor(self, number_of_state_data_sets):
        """"""

    @abstractmethod
    def _normalize(self, residual):
        """"""

    @abstractmethod
    def _calculate(self, normalized_residual):
        """"""

    @property
    @abstractmethod
    def name(self):
        """"""

    def __call__(self, residuals):
        return self.normalize_and_calculate(residuals)


class NormMetricFunction(_MetricFunctionBase):
    """
    This is the general norm metric function. It is a wrapper for NumPy's
    ``linalg.norm`` function. Any order valid for vectors with the 
    ``linalg.norm`` function 
    can be used to initialize this metric function.

    This metric function is only used to form an objective 
    for algorithms that are not least squares algorithms. However, 
    it does provide normalization information that will act on 
    the residuals for least squares algorithms. 

    .. note::
       Least squares algorithms should use the 
       :class:`~matcal.core.objective.L2NormMetricFunction` unless
       specific behavior is desired.

    .. note::
        Normalization due to the number of data sets in a state
        and the length of the data are not applied when using an 
        order of 0. They also are not relevant when using the infinity
        norms. 

    .. warning:: 
        The order 0 and infinity norms will likely not perform well
        when using multiple states, models or data sets. They also
        may not to perform well in gradient calibrations. 
    """
    valid_orders = [None, np.inf, -np.inf, 0, 1, -1, 2, -2]

    class OrderError(RuntimeError):
        pass

    def __init__(self, order=None):
        self._check_order_input(order)
        self._order = order
        self._normalizer = self._assign_normalizer()

    def _check_order_input(self, order):
        if order not in self.valid_orders:
            valid_orders_string = ""
            for ord in self.valid_orders:
                valid_orders_string += f"{ord}\n"
            error_str = "Incorrect order passed to the NormMetricFunction init." + \
                        f" Check input. Valid orders are: \n{valid_orders_string}"
            raise self.OrderError(error_str)

    def _assign_normalizer(self):
        if self._order is None or self._order == 2:
            return SqrtDataSizeNormalizer()
        elif self._order == -2:
            return InvertedSqrtDataSizeNormalizer()
        elif self._order == 1:
            return LinearDataSizeNormalizer()
        elif self._order == -1:
            return InvertedLinearDataSizeNormalizer()
        else:
            return None
    
    def _calculate(self, residual):
        return np.linalg.norm(residual, self._order)

    def _normalize(self, residual):
        if self._normalizer:
            return self._normalizer.apply(residual)
        else:
            return residual

    def calculate_group_normalization_factor(self, number_of_state_data_sets):
        if self._order == 2 or self._order is None:
            return 1 / np.sqrt(number_of_state_data_sets)
        elif self._order == -2:
            return np.sqrt(number_of_state_data_sets)
        elif self._order == 1:
            return 1 / number_of_state_data_sets
        elif self._order == -1:
            return number_of_state_data_sets
        
        else:
            return 1

    @property
    def name(self):
        """
        :return: The name of the metric method used.
        """
        return f"L{self._order} norm"


class L2NormMetricFunction(NormMetricFunction):
    """
    Produces the :math:`l_2` Norm of the residual for the objective. 
    The :math:`l_2` Norm is calculated using

    .. math:: l_2=\\sqrt{\\sum_i \\left(r_i\\right)^2}

    where :math:`\\mathbf{r}` is the residual for this objective.
    """

    def __init__(self):
        super().__init__(2)


class L1NormMetricFunction(NormMetricFunction):
    """
    Produces the :math:`l_1` Norm of the residual for the objective. 
    The :math:`l_1` Norm is calculated using

    .. math:: l_1=\\sum_i \\left|r_i\\right|

    where :math:`\\mathbf{r}` is the residual for this objective.
    """

    def __init__(self):
        super().__init__(1)


class SumSquaresMetricFunction(L2NormMetricFunction):
    """
    Produces the square of the :math:`l_2` norm of the residual, otherwise known as 
    the sum of squares error (SSE). it is calculated using

    .. math:: SSE=\\sum_i \\left(r_i\\right)^2

    where :math:`\\mathbf{r}` is the residual for this objective.
    """
    @property
    def name(self):
        """
        :return: The name of the metric method used.
        """
        return "SSE"

    def _calculate(self, residual):
        """
        Calculates :math:`l_2^2` norm of the residual vector.

        :param residual: The residual vector to be used for the error metric calculation.
        :type residual: array

        :return: Returns the SSE (  :math:`l_2^2` ) of the residual.
        """

        return super()._calculate(residual)**2


class _ObjectiveBase(ABC):
    _id_numbers = count(0)

    class TypeError(RuntimeError):
        pass

    class MissingFieldsError(RuntimeError):
        pass

    class InconsistentDataError(RuntimeError):
        pass

    def __init__(self, metric_function, field_weights, fields_of_interest):

        self._residual_calculator = NominalResidualCalculator(*fields_of_interest)
        self._required_fields = list(self.fields_of_interest)
        self._field_weights = None
        self._simulation_qoi_extractor = ReturnPassedDataExtractor()
        self._experiment_qoi_extractor = ReturnPassedDataExtractor()
        self._metric_function = None
        self._name = None

        self.set_metric_function(metric_function)
        self.set_field_weights(field_weights)
        self._id_number = next(self._id_numbers)
        self.set_name(self._get_unique_name())
        self._independent_field = None
        self._large_data_sets = False

    def _check_required_fields_are_in_data(self, data, required_fields):
        passed_field_names = data.field_names
        fields_not_in_data = self._determine_required_fields_not_in_data(
            passed_field_names, required_fields)

        if fields_not_in_data:
            error_string = self._create_required_fields_not_in_data_error_string(
                data, fields_not_in_data)
            raise self.MissingFieldsError(error_string)

    def _determine_required_fields_not_in_data(self, passed_field_names, required_fields):
        fields_not_in_data = []
        for field in required_fields:
            if field not in passed_field_names:
                fields_not_in_data.append(field)
        return fields_not_in_data

    def _create_required_fields_not_in_data_error_string(self, data, fields_not_in_data):
        fields_in_data_string = self._create_character_delimited_string_from_list_of_strings(
            data.field_names)
        fields_not_in_data_string = self._create_character_delimited_string_from_list_of_strings(
            fields_not_in_data)
        error_string = ("\n\nThe following required fields were not"
                        f"in the passed data set \"{data.name}\":\n")
        error_string += fields_not_in_data_string
        error_string += "\nIt does have the following fields:\n" + fields_in_data_string
        return error_string

    def _create_character_delimited_string_from_list_of_strings(self, strings, delimiter='\n'):
        combined_string = ""
        for string in strings:
            combined_string += string + delimiter
        return combined_string

    @abstractmethod
    def calculate(self, conditioned_experiment_data_collection, experiment_data_collection,
                  conditioned_simulation_data_collection, simulation_data_collection):
        """"""

    @abstractmethod
    def _get_unique_name(self):
        """"""

    @property
    @abstractmethod
    def _class_name(self):
        """"""

    @property
    def name(self):
        return self._name

    @property
    def fields_of_interest(self):
        return self._residual_calculator.fields_of_interest

    @property
    def experiment_qoi_extractor(self):
        return self._experiment_qoi_extractor

    @property
    def simulation_qoi_extractor(self):
        return self._simulation_qoi_extractor

    @property
    def field_weights(self):
        return self._field_weights
    
    @property
    def independent_field(self):
        return self._independent_field

    @property
    def large_data_sets(self):
        return self._large_data_sets

    def set_as_large_data_sets_objective(self, large_data_sets=True):
        """
        Identify the objective as working with large data sets.
        If this is used, the data saved and written to file is reduced. 
        The simulation and experimental data are not saved. 
        Also, the conditioned simulation and experimental QoIs and 
        the conditioned and weighted residuals
        are not saved.

        :param large_data_sets: set the value as True or False. Can be used to 
            allow saving data for default large data set objectives if set to 
            False.
        :type large_data_sets: bool
        """
        self._large_data_sets = large_data_sets

    def set_metric_function(self, metric_function):
        """
        Set the metric function to be used. It must be a valid 
        metric function from :mod:`~matcal.core.objective`.
        
        :param metric_function: the desired metric function to be used. 
        """
        self._check_item_is_correct_type(
            metric_function, _MetricFunctionBase, "metric function")

        self._metric_function = metric_function

    def use_log_residual(self):
        """
        Calculate the residual by subtracting the natural log of the QOI's. 
        This can be useful when dealing with large ranges of values.
        Currently, this method requires that all QOI values be positive. 
        """
        self._residual_calculator = LogResidualCalculator(*self.fields_of_interest)


    def set_field_weights(self, *field_weights):
        """
        :param field_weights: the desired weights to be applied to 
            each objective. Note that this will be applied to
            every state that is being evaluated for this objective. 
            The states are defined by the data collection
            or state collection passed to the study evaluation set that the 
            objective is part of.
        :type field_weights: list(:class:`~matcal.core.residuals.ResidualWeightingBase`)
        """
        for field_weight in field_weights:
            self._check_item_is_correct_type(
                field_weight, ResidualWeightingBase, "field weight")
        self._field_weights = field_weights

    def set_qoi_extractors(self, qoi_extractor):
        """
        Sets the QoI extractor that will be applied to both the experiment and simulation data.

        :param qoi_extractor: A valid QoI extractor from :mod:`~matcal.core.qoi_extractor`

        """
        self._check_item_is_correct_type(
            qoi_extractor, QoIExtractorBase, "QoI extractors")
        self._simulation_qoi_extractor = qoi_extractor
        self._experiment_qoi_extractor = qoi_extractor

    def set_simulation_qoi_extractor(self, qoi_extractor):
        """
        Sets the QoI extractor that will be applied to only the simulation data.

        :param qoi_extractor: A valid QoI extractor from :mod:`~matcal.core.qoi_extractor`

        """
        self._check_item_is_correct_type(
            qoi_extractor, QoIExtractorBase, "simulation QoI extractor")
        self._simulation_qoi_extractor = qoi_extractor

    def set_experiment_qoi_extractor(self, qoi_extractor):
        """
        Sets the QoI extractor that will be applied to only the experimental data.

        :param qoi_extractor: A valid QoI extractor from :mod:`~matcal.core.qoi_extractor`

        """
        self._check_item_is_correct_type(
            qoi_extractor, QoIExtractorBase, "experiment QoI extractor")
        self._experiment_qoi_extractor = qoi_extractor

    def set_name(self, name):
        """
        Sets the name for the objective which is used to extract the data from the study results. 
        A default name is applied based on the order it is added to an evaluation set.

        :param qoi_extractor: A valid QoI extractor from :mod:`~matcal.core.qoi_extractor`

        """
        self._check_item_is_correct_type(name, str, "objective name")
        self._name = name

    def data_specific_initialization(self, exp_data_collection):
        pass

    def has_fields_of_interest(self):
        return self._residual_calculator.fields_of_interest is not None

    def _check_item_is_correct_type(self, item, desired_type, message):
        if not isinstance(item, desired_type):
            raise self.TypeError(
                "The objective expected a {} type for the added {}, object of type {}".format(
                    desired_type, message, type(item)))

    def has_independent_field(self):
        """
        Returns true if the objective has an independent field. An independent field
        is required for some studies.
        """
        return self._independent_field is not None


class Objective(_ObjectiveBase):
    _class_name = "Objective"

    def __init__(self, *fields_of_interest):
        """
        The Objective class handles the calculation of the residual vector and merit functions when
        comparing sets of 1D vector for a study evaluation set.
        See :meth:`~matcal.core.parameter_studies.ParameterStudy.add_evaluation_set`.

        The user must specify the fields of interest for the objective. 
        The residual vector is calculated by subtracting
        the experimental data from the simulation data and assumes that 
        this subtraction is as desired. If either
        data set must be manipulated before the comparison use the
        :meth:`~matcal.core.objective.Objective.set_qoi_extractors`,
        :meth:`~matcal.core.objective.Objective.set_simulation_qoi_extractor` or
        :meth:`~matcal.core.objective.Objective.set_experiment_qoi_extractor` to do so.

        :param fields_of_interest: the fields to be evaluated in the objective
        :type fields_of_interest: list(str)

        :raises Objective.TypeError: If the wrong types are passed into the constructor.
        """
        self._check_fields_of_interest_are_correct_type(fields_of_interest)
        metric_function = SumSquaresMetricFunction()
        field_weights = IdentityWeighting()
        super().__init__(metric_function, field_weights, fields_of_interest)
        self._qois = None
        self._results = None

    def _get_unique_name(self):
        name_list = [self._class_name, 
                     str(self._id_number)]
        return "_".join(name_list)
    
    def calculate(self, conditioned_experiment_qoi_collection, experiment_qoi_collection,
                  conditioned_simulation_qoi_collection, simulation_qoi_collection):
        self._qois, self._results = self._initialize_containers()
        self._set_data_collections_in_objective_results(conditioned_experiment_qoi_collection,
                                                        experiment_qoi_collection,
                                                        conditioned_simulation_qoi_collection, 
                                                        simulation_qoi_collection)
        for state in conditioned_experiment_qoi_collection.states.values():
            self._calculate_state_residual(state)
        self._finalize_results()
        return self._results, self._qois

    def _initialize_containers(self):
        qoi = ObjectiveQOI(self._required_fields, self.fields_of_interest, self.large_data_sets)
        results = ObjectiveResults(self._required_fields, self.fields_of_interest, self.large_data_sets)
        return qoi, results
                
    def _calculate_state_residual(self, state):
        zipped_qois = zip(self._qois.experiment_qois[state.name],
                          self._qois.simulation_qois[state.name], 
                          self._qois.conditioned_experiment_qois[state.name],
                          self._qois.conditioned_simulation_qois[state.name])
        logger.debug(f"Retrieved exp qois length {len(self._qois.experiment_qois[state.name])}")
        logger.debug(f"Retrieved sim qois length {len(self._qois.simulation_qois[state.name])}")

        for qoi_index, qois in enumerate(zipped_qois):
            (exp_qoi, sim_qoi, conditioned_exp_qoi, conditioned_sim_qoi) = qois
            self._confirm_experiment_fields(exp_qoi)
            self._confirm_simulation_fields(sim_qoi)
            try:
                residual = self._residual_calculator.calculate(exp_qoi, sim_qoi)
                conditioned_residual = self._residual_calculator.calculate(conditioned_exp_qoi,
                                                                            conditioned_sim_qoi)
            except NominalResidualCalculator.InconsistentDataError:
                raise self.InconsistentDataError(
                    f"The objective \"{self.name}\" could not compare the"
                    f" two data sets for state \"{state.name}\" due to inconsistent data length.")

            self._results.add_residuals(residual)
            self._results.add_conditioned_residuals(conditioned_residual)
            self._apply_simulation_and_residual_weighting(
                state, qoi_index)
            self._results.add_weighted_conditioned_objective(
                self._convert_residuals_to_objectives(state))

            self._apply_experiment_weighting(state, qoi_index)
            logger.debug(f"Objective qoi index for state \"{state.name}\" is {qoi_index}")

            for data in self._results.weighted_conditioned_residuals[state.name]:
                for field in data.field_names:
                    logger.debug(f"Residual length for state \"{state.name}\" "
                                 f"for field \"{field}\" is {len(data[field])}")

    def _convert_residuals_to_objectives(self, state):
        residual = self._results.weighted_conditioned_residuals[state.name][-1]
        objectives = {}
        for field in residual.field_names:
            objective_value = self._metric_function.normalize_and_calculate(
                residual[field])
            logger.debug(f"Residual length for field \"{field}\" is {len(residual)}\n")
            objectives[field] = objective_value
        objectives = convert_dictionary_to_data(objectives)
        objectives.set_state(state)
        return objectives

    def _set_data_collections_in_objective_results(self, conditioned_experiment_qoi_collection,
                                                    experiment_qoi_collection,
                                                   conditioned_simulation_qoi_collection, 
                                                   simulation_qoi_collection):
        self._qois.set_conditioned_experiment_qois(
            conditioned_experiment_qoi_collection)
        self._qois.set_conditioned_simulation_qois(
            conditioned_simulation_qoi_collection)
        self._qois.set_experiment_qois(experiment_qoi_collection)
        self._qois.set_simulation_qois(simulation_qoi_collection)

    def _finalize_results(self):
        norm_func = self._metric_function.normalize_only
        group_norm_func = self._metric_function.calculate_group_normalization_factor

        normalized_residuals = _normalize_residuals(
            self._results._weighted_conditioned_residuals, norm_func,
            group_norm_func)
        self._results.set_weighted_conditioned_normalized_residuals(normalized_residuals)
        obj = self._metric_function.calculate_only(self._results.calibration_residuals)
        self._results.set_objective(obj)
        self._results.reset_qois_for_large_data_sets()
        self._qois.reset_qois_for_large_data_sets()

    def _confirm_simulation_fields(self, simulation_data):
        self._check_required_fields_are_in_data(
            simulation_data, self._required_fields)

    def _confirm_experiment_fields(self, exp_data):
        self._check_required_fields_are_in_data(
            exp_data, self._required_fields)

    def _apply_experiment_weighting(self, state, qoi_index):
        cond_exp_qois = self._qois.conditioned_experiment_qois[state.name][qoi_index]
        exp_qois = self._qois.experiment_qois[state.name][qoi_index]
        weighted_conditioned_exp_qois = self.apply_weights(cond_exp_qois,
                                                           exp_qois, 
                                                           self._field_weights,
                                                           cond_exp_qois)

        self._qois.add_weighted_conditioned_experiment_qois(
            weighted_conditioned_exp_qois)

    def _apply_simulation_and_residual_weighting(self, state, qoi_index):
        cond_exp_qois = self._qois.conditioned_experiment_qois[state.name][qoi_index]
        exp_qois = self._qois.experiment_qois[state.name][qoi_index]
        cond_resids = self._results.conditioned_residuals[state.name][-1]
        weighted_conditioned_residuals = self.apply_weights(cond_exp_qois,
                                                             exp_qois, 
                                                             self._field_weights,
                                                             cond_resids)
        resids = self._results._residuals[state.name][-1]
        weighted_residuals = self.apply_weights(cond_exp_qois,
                                                exp_qois, 
                                                self._field_weights,
                                                resids)

        cond_sim_qois = self._qois.conditioned_simulation_qois[state.name][qoi_index]
        weighted_conditioned_sim_qois = self.apply_weights(cond_exp_qois,
                                                           exp_qois, 
                                                           self._field_weights,
                                                           cond_sim_qois)

        self._results.add_weighted_residuals(
            weighted_residuals)

        self._results.add_weighted_conditioned_residuals(
            weighted_conditioned_residuals)
        self._qois.add_weighted_conditioned_simulation_qois(
            weighted_conditioned_sim_qois)

    @staticmethod
    def apply_weights(conditioned_data, data, weight_list, residual):
        weighted_residual = residual
        for weighting in weight_list:
            weighted_residual = weighting.apply(
                conditioned_data, data, weighted_residual)
        return weighted_residual

    def _check_fields_of_interest_are_correct_type(self, fields):
        for field in fields:
            self._check_item_is_correct_type(field, str, "field of interest")


class CurveBasedInterpolatedObjective(Objective):
    """
    The CurvedBasedInterpolatedObjective class handles the calculation of the residual 
    vector and merit functions when comparing sets of 2D curves for for a study evaluation set.
    This objective will attempt to normalize data set size and scale to remove any implicit bias
    in the data from these factors.
    See :meth:`~matcal.core.parameter_studies.ParameterStudy.add_evaluation_set`.
    """
    _class_name = "CurveBasedInterpolatedObjective"

    def __init__(self, independent_field, *dependent_fields, left=None, right=None, period=None):
        """
        Specify the fields of interest for the objective. Generally, 
        when comparing data, there is some independent
        field (e.g. time, displacement, etc.) at which to compare some 
        dependent field (load, temperature, etc.).
        This method allows the user to specify these fields for the objective. 
        By default, this objective will
        compare the simulation and experiment dependent fields at the experiment 
        independent field values. To do so,
        it will interpolate the simulation data to the experimental data 
        independent field values. 
        It does not extrapolate in either direction (independent variable 
        values less than the least simulation output or greater 
        than the greatest simulation output) and handles values outside the 
        simulation limits as is done by NumPy interp.

        :param independent_field: the name of the independent field.
        :type independent_field: str

        :param dependent_fields: the dependent fields for the objective.
        :type dependent_fields: list(str)

        :param left: the left parameter as described in the NumPy interp function. 
        :type left: float

        :param right: the right parameter as described in the NumPy interp function
        :type right: float

        :param period: the period parameter as described in the NumPy interp function
        :type period: float

        :raises Objective.TypeError: If the wrong types are passed into the constructor.
        """
        super().__init__(*dependent_fields)
        self._check_fields_are_correct_type(
            independent_field, dependent_fields)
        self._independent_field = independent_field
        self._check_numpy_interp_parameters(left=left, right=right, period=period)
        self.set_simulation_qoi_extractor(
            InterpolatingExtractor(self._independent_field, left=left, right=right, period=period))
        self._required_fields.append(independent_field)

    def _check_fields_are_correct_type(self, independent_field, dependent_fields):
        self._check_item_is_correct_type(
            independent_field, str, "independent field")
        for dependent_field in dependent_fields:
            self._check_item_is_correct_type(
                dependent_field, str, "dependent field")

    def _check_numpy_interp_parameters(self, **kwargs):
        for key, val in kwargs.items():
            if val is not None:
                if not isinstance(val, numbers.Real):
                    raise self.TypeError("The CurveBasedInterpolatedObjective parameter "
                                         f"\"{key}\" must be numeric. Received a "
                                         f"value of type \"{type(val)}\".")


# Testing for this class is based off of the CurveBasedInterpolatedObjective
# this is because it inherits most of the functionality of that class.
# if this class changes more it may warrant more testing. 
class DirectCurveBasedInterpolatedObjective(CurveBasedInterpolatedObjective):
    """
    The DirectCurveBasedInterpolatedObjective class handles the calculation of the residual 
    vector and merit functions when comparing sets of 2D curves for for a study evaluation set.
    This objective formulation does not do any adjustments of the residual.
    See :meth:`~matcal.core.parameter_studies.ParameterStudy.add_evaluation_set`.
    """
    _class_name = "DirectCurveBasedInterpolatedObjective"
       
    def _pass_through(self, data):
        return data
    
    def _calc_unity(self, *args):
        return 1.
    
    def _finalize_results(self):
        normalized_residuals = _normalize_residuals(
            self._results._residuals, self._pass_through, self._calc_unity)
        self._results.set_weighted_conditioned_normalized_residuals(normalized_residuals)
        obj = self._metric_function.calculate_only(self._results.calibration_residuals)
        self._results.set_objective(obj) 
        
    def set_field_weights(self, *field_weights):
        if len(field_weights) > 1 or not isinstance(field_weights[0], IdentityWeighting):
            message = "Direct objectives do not accept field weights. "
            message += "Please use an objective with out the prefix 'Direct'"
            raise RuntimeError(message)
        else:
            super().set_field_weights(*field_weights)


class SimulationResultsSynchronizer(DirectCurveBasedInterpolatedObjective):
    _class_name = "SimulationResultsSynchronizer"
        
    def __init__(self, independent_field, independent_field_values, 
                 *dependent_fields, 
                  left=None, right=None, period=None):
        """
        This objective will the simulation values 
        for the dependent fields at the user specified
        independent field values. It is not used to compare 
        data to other data sources and can be useful for 
        sensitivity and parameter studies.
        It will interpolate the simulation data to the user specified 
        independent field values. 
        It does not extrapolate in either direction (independent variable 
        values less than the least simulation output or greater 
        than the greatest simulation output) and handles values outside the 
        simulation limits as is done by NumPy interp.
        In a sense, it is still an "objective" because it is comparing the 
        data to vectors of zero, but also "synchronizes" simulation results
        to common independent field data locations. 

        No data normalization or conditioning is provided 
        for this objective by default and weighting is not available.
        
        .. note::
            The metric function for this objective is the 
            :class:`~matcal.core.objective.L2NormMetricFunction` which 
            is more relevant to sensitivity and parameter studies 
            for which this objective was intended.

        :param independent_field: the name of the independent field.
        :type independent_field: str

        :param independent_field_values: the values of the independent field.
        :type independent_field_values: ArrayLike

        :param dependent_fields: the dependent fields for the objective.
        :type dependent_fields: list(str)

        :param left: the left parameter as described in the NumPy interp function. 
        :type left: float

        :param right: the right parameter as described in the NumPy interp function
        :type right: float

        :param period: the period parameter as described in the NumPy interp function
        :type period: float

        :raises Objective.TypeError: If the wrong types are passed into the constructor.
        """
        super().__init__(independent_field, *dependent_fields, 
                       left=left, right=right, period=period)
        
        self._independent_field_values = self._check_independent_field_values(
            independent_field_values)
        self.set_metric_function(L2NormMetricFunction())

    def _check_independent_field_values(self, independent_field_values):
        if isinstance(independent_field_values, numbers.Real):
            return np.atleast_1d(independent_field_values)
        else:
            check_value_is_array_like_of_reals(independent_field_values, 
                                               "independent_field_values", 
                                               "SensitivityObjective")
            return independent_field_values
            
    def _generate_experimental_data_qois(self, state):
        exp_data_dict = OrderedDict()
        for field in self.fields_of_interest:
            exp_data_dict[field] = np.zeros(len(self._independent_field_values))
        exp_data_dict[self._independent_field] = self._independent_field_values
        exp_data = convert_dictionary_to_data(exp_data_dict)
        exp_data.set_state(state)
        return exp_data


class ObjectiveCollection(CollectionBase):
    """
      A collection of :class:`~matcal.core.objective.Objective` objects. 
      This is used to combine multiple objective
      objects so that they can be passed to a MatCal study evaluation set.
      """
    _collection_type = _ObjectiveBase

    def __init__(self, name, *objectives):
        """
        :param name: The name of the objective collection.
        :type name: str

        :param objectives: the objectives to be added to the collection.
        :type objectives: list(:class:`~matcal.core.objective.Objective`)

        :raises CollectionValueError: If name is a an empty string.
        :raises CollectionTypeError: If name is not a string and the parameters 
            to be added to the collection are not of
            the correct type.
        """
        super().__init__(name, *objectives)


class _ObjectiveSetBase(ABC):
    _id_numbers = count(0)

    class TypeError(RuntimeError):
        pass

    def __init__(self, objective_collection, data_collection, state_collection, 
                 conditioner_class=RangeDataConditioner):
        self._conditioner_class=conditioner_class
        self._id_number = next(self._id_numbers)
        self.name = "objective_set_{}".format(self._id_number)
        self._objective_collection = None

        self._check_item_is_correct_type(
            state_collection, StateCollection, "state collection")
        self._state_collection = state_collection

        self._check_item_is_correct_type(
            data_collection, DataCollection, "data collection")
        self._experiment_data_collection = data_collection
        self._experiment_qoi_collections_by_obj = OrderedDict()
        self._data_conditioners = OrderedDict()
        self._conditioned_exp_qoi_collections_by_obj = OrderedDict()
        self._number_of_objectives = None
        self._residual_vector_length = None
        self._flattened_weighted_conditioned_experiment_qois = np.array([], dtype=np.double)
        self._objective_set_results = OrderedDict()

        self.update_objective_collection(objective_collection)

    def _check_item_is_correct_type(self, item, desired_type, message):
        if not isinstance(item, desired_type):
            raise self.TypeError(
                f"The objective set expected a {desired_type} type for"
                 f" the added {message}, object is of type {type(item)}")

    def _get_residual_length_from_results(self, results):
        return len(results.calibration_residuals)

    def get_objective_names(self):
        return list(x.name for x in self._objective_collection.values())

    def update_objective_collection(self, objective_collection):
        self._check_item_is_correct_type(
            objective_collection, ObjectiveCollection, "objective collection")
        self._objective_collection = objective_collection
        self._experiment_qoi_collections_by_obj = self._initialize_experiment_qois()
        self._conditioned_exp_qoi_collections_by_obj = self._create_conditioned_experiment_qois()
        self._initialize_objectives_length_residuals_length_and_flattened_exp_qois()
        
    def _initialize_objectives_length_residuals_length_and_flattened_exp_qois(self):
        residual_vector_length = 0
        number_of_objectives = 0
        self._flattened_weighted_conditioned_experiment_qois = np.array([], dtype=np.double)
        for objective in self._objective_collection.values():
            for state in self._state_collection.values():
                exp_qoi_collection = (
                    self._experiment_qoi_collections_by_obj[objective.name][state])
                for qoi_index, exp_qois in enumerate(exp_qoi_collection):
                    conditioned_exp_qois = (
                        self._conditioned_exp_qoi_collections_by_obj[objective.name]
                        [state][qoi_index])
                    weighted_conditioned_extracted_qois = (
                        objective.apply_weights(conditioned_exp_qois,
                                                exp_qois, objective._field_weights,
                                                conditioned_exp_qois))
                    for field in objective.fields_of_interest:
                        self._add_exp_qois_to_flattened_exp_qois(
                            weighted_conditioned_extracted_qois[field])
                        residual_vector_length += conditioned_exp_qois[field].length
            number_of_objectives += 1

        self._number_of_objectives = number_of_objectives
        self._residual_vector_length = residual_vector_length

    def _add_exp_qois_to_flattened_exp_qois(self, weighted_conditioned_exp_field_qois):
        self._flattened_weighted_conditioned_experiment_qois = (
            np.append(self._flattened_weighted_conditioned_experiment_qois,
                      weighted_conditioned_exp_field_qois))

    def get_flattened_weighted_conditioned_experiment_qois(self):
        return self._flattened_weighted_conditioned_experiment_qois

    @property
    def residual_vector_length(self):
        return self._residual_vector_length

    @property
    def number_of_objectives(self):
        return self._number_of_objectives

    @property
    def conditioned_experiment_qoi_collection(self):
        return self._conditioned_exp_qoi_collections_by_obj

    @property
    def states(self):
        return self._state_collection

    @property
    def objectives(self):
        return self._objective_collection

    @property
    def data_collection(self):
        return self._experiment_data_collection

    @abstractmethod
    def _create_conditioned_experiment_qois(self):
        """"""

    @abstractmethod
    def condition_data_collection(self, simulation_data_collection, objective):
        """"""

    def calculate_objective_set_results(self, simulation_data_collection):
        self._objective_set_results = OrderedDict()
        self._objective_set_qois = OrderedDict()
        for objective in self._objective_collection.values():
            new_obj_result_name = objective.name
            results, qois = self._get_this_objectives_results(objective, simulation_data_collection)
            qois.set_experiment_data(self._experiment_data_collection)
            qois.set_simulation_data(simulation_data_collection)
            conditioned_sim_data = self.condition_data_collection(simulation_data_collection, 
                                                                  objective)
            qois.set_conditioned_simulation_data(conditioned_sim_data)
            conditioned_exp_data = self.condition_data_collection(self._experiment_data_collection, 
                                                                  objective)
            qois.set_conditioned_experiment_data(conditioned_exp_data)
            self._objective_set_results[new_obj_result_name] = results
            self._objective_set_qois[new_obj_result_name] = qois
        
        del simulation_data_collection
        
        return self._objective_set_results, self._objective_set_qois

    def _get_this_objectives_results(self, objective, sim_data_collection):
        simulation_qois = self._extract_simulation_qois(sim_data_collection, objective)
        exp_qois = self._retrieve_experiment_qois(objective)
        conditioned_simulation_qois = self.condition_data_collection(simulation_qois, objective)
        conditioned_exp_qois = self.condition_data_collection(exp_qois, objective)
        results, processed_qois = objective.calculate(conditioned_exp_qois, exp_qois,
                                      conditioned_simulation_qois, simulation_qois)

        return results, processed_qois

    def _extract_simulation_qois(self, simulation_data_collection, objective):
        return self._extract_qois(simulation_data_collection, self._experiment_data_collection, 
                                  objective.simulation_qoi_extractor, objective.fields_of_interest)
        
    def _extract_experiment_qois(self, ref_data_collection, objective):
        return self._extract_qois(self._experiment_data_collection, ref_data_collection, 
                                  objective.experiment_qoi_extractor, objective.fields_of_interest)

    def _retrieve_experiment_qois(self, objective):
        return self._experiment_qoi_collections_by_obj[objective.name]
        
    def _extract_qois(self, working_data_collection, reference_data_collection, extractor, fields):
        qoi_collection = DataCollection("qois " + working_data_collection.name)
        for state in self._state_collection.values():
            for reference_data in reference_data_collection[state.name]:
                logger.debug(f"Adding {reference_data.name} data to qois")
                for working_data in working_data_collection[state.name]:
                    qoi_data = extractor.calculate(working_data, reference_data, fields)
                    qoi_data.set_state(state)
                    qoi_collection.add(qoi_data)
        return qoi_collection

    def _initialize_experiment_qois(self):
        initial_exp_qois = OrderedDict()
        for objective in self._objective_collection.values():
            objective.data_specific_initialization(self._experiment_data_collection)
            fake_simulation_data = self._make_fake_simulation_data(objective)
            initial_exp_qois[objective.name] = self._extract_experiment_qois(fake_simulation_data, 
                                                                             objective)
            objective.experiment_qoi_extractor.clean_up()

        return initial_exp_qois

    def _make_fake_simulation_data(self, objective):
        one_data_exp_dc = DataCollection("one data exp dc")
        for state in self._experiment_data_collection.states:
            one_data_exp_dc.add(self._experiment_data_collection[state][0])
        return one_data_exp_dc

    def _get_experiment_qois(self, ref_data_collection):
        exp_qois = OrderedDict()
        for objective in self._objective_collection.values():
            exp_qois[objective.name] = self._extract_experiment_qois(ref_data_collection, objective)
        return exp_qois

    def purge_unused_data(self):
        new_dc = DataCollection(f'Pruned:{self.data_collection.name}')
        required_fields = self._identify_required_experimental_data_fields()
        for state, data_list in self._experiment_data_collection.items():
            for current_data in data_list:
                new_data = self._prune_data(required_fields, current_data)
                if new_data is not None:
                    new_dc.add(new_data)       
        self._replace_data(new_dc)

    def _identify_required_experimental_data_fields(self):
        required_fields = []
        for objective in self.objectives.values():
            for f in objective._experiment_qoi_extractor.required_experimental_data_fields:
                required_fields.append(f)
            for f in objective._simulation_qoi_extractor.required_experimental_data_fields:
                required_fields.append(f)
            required_fields += objective.fields_of_interest
        return required_fields

    def _prune_data(self, required_fields, current_data):
        new_data = current_data.copy()
        for var_name in current_data.field_names:
            if var_name not in required_fields:
                new_data = new_data.remove_field(var_name)   
        return new_data

    def _replace_data(self, new_dc):
        del self._experiment_data_collection
        self._experiment_data_collection = new_dc


class ObjectiveSet(_ObjectiveSetBase):
    """
    Not intended for users. A set of data and objectives to be calculated for
    a model for a given state 
    collection.
    """
    def _create_conditioned_experiment_qois(self):
        conditioned_qois = OrderedDict()
        for objective in self._objective_collection.values():
            self._data_conditioners[objective.name] = OrderedDict()
            exp_data_qois = self._experiment_qoi_collections_by_obj[objective.name]
            conditioned_qois[objective.name] = DataCollection("conditioned experiment qois")
            for state in self._state_collection.values():
                qoi_list = exp_data_qois[state.name]
                self.initialize_state_conditioner(qoi_list, state, objective)
                for qoi_item in qoi_list:
                    conditioned_qoi_item = \
                        self._data_conditioners[objective.name][state.name].apply_to_data(qoi_item)
                    conditioned_qois[objective.name].add(conditioned_qoi_item)
        return conditioned_qois

    def initialize_state_conditioner(self, data_list, state, objective):
        state_conditioner = self._conditioner_class()
        state_conditioner.initialize_data_conditioning_values(data_list)
        self._data_conditioners[objective.name][state.name] = state_conditioner

    def condition_data_collection(self, data_collection, objective):
        conditioned_data_collection = DataCollection("conditioned "+data_collection.name)
        for state_name in self._state_collection.keys():
            for data in data_collection[state_name]:
                data_conditioner = self._data_conditioners[objective.name][state_name]
                conditioned_data = data_conditioner.apply_to_data(data)
                conditioned_data_collection.add(conditioned_data)
        return conditioned_data_collection


def _normalize_residuals(residuals_dc, norm_function, group_norm_function):
    dc_name = "weighted conditioned normalized residuals"
    weighted_conditioned_normalized_residuals = DataCollection(dc_name)
    state_list = residuals_dc.keys()
    state_number_factor = group_norm_function(len(state_list))
    for state in state_list:
        residual_list = residuals_dc[state]
        repeat_factor = group_norm_function(len(residual_list))
        for residuals in residual_list:
            normalized_resids = deepcopy(residuals)
            for field_name in residuals.keys():
                normalized_resids[field_name] = (norm_function(residuals[field_name]) * 
                                                 state_number_factor * repeat_factor)
            weighted_conditioned_normalized_residuals.add(normalized_resids)
    return weighted_conditioned_normalized_residuals
