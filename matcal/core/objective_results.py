"""
This module only contains the ObjectiveResults class which is a large data 
structure for holding all data related to objective calculation.
"""

from abc import abstractmethod, ABC
import numpy as np

from matcal.core.data import DataCollection


def flatten_data_collection(data_collection, fields_of_interest=None):
    data_array = np.array([])
    for state in data_collection.keys():
        for data in data_collection[state]:
            if fields_of_interest is None:
                fields_of_interest = data.field_names
            for field in fields_of_interest:
                data_array = np.append(data_array, np.atleast_1d(data[field]))
    return data_array 


class ObjectiveContainerBase(ABC):

    def __init__(self, required_fields, fields_of_interest, large_data_sets=False):
        self._required_fields = required_fields
        self._fields_of_interest = fields_of_interest
        self._large_data_sets = large_data_sets
       
    def flatten_data_collection(self, data_collection):
        return flatten_data_collection(data_collection, self._fields_of_interest)

    def _add_data_to_data_collection(self, data_collection, data, required_fields):
        new_data = self._remove_unneeded_fields_from_data(data, required_fields)
        data_collection.add(new_data)
    
    class InvalidResidualError(RuntimeError):

        def __init__(self, bad_residual):
            message = "ObjectiveResults received an invalid residuals result. "
            message += "Residuals must be a Datacollection."
            message += f"\n Received: Type: {type(bad_residual)}"
            super().__init__(message)

    @staticmethod
    def _remove_unneeded_fields_from_data(data, required_fields):
        return data[list(required_fields)]

    def reset_qois_for_large_data_sets(self):
        if self._large_data_sets:
            self._initialize_nonessential_qois()

    @abstractmethod
    def _initialize_nonessential_qois(self):
        """"""

class ObjectiveQOI(ObjectiveContainerBase):

    def __init__(self, required_fields, fields_of_interest, large_data_sets=False):
        super().__init__(required_fields, fields_of_interest, large_data_sets)

        self._experiment_qois = DataCollection("experiment qois")
        self._simulation_qois = DataCollection("simulation qois")

        self._experiment_data = None
        self._simulation_data = None

        self._conditioned_experiment_data = None
        self._conditioned_simulation_data = None

        self._conditioned_experiment_qois = None
        self._conditioned_simulation_qois = None

        self._weighted_conditioned_experiment_qois = None
        self._weighted_conditioned_simulation_qois = None

        self._initialize_nonessential_qois()

    def _initialize_nonessential_qois(self):
        self._experiment_data = DataCollection("experiment data")
        self._simulation_data = DataCollection("experiment data")

        self._conditioned_experiment_data = DataCollection("conditioned experiment data")
        self._conditioned_simulation_data = DataCollection("conditioned simulation data")

        self._conditioned_experiment_qois = DataCollection("conditioned experiment qois")
        self._conditioned_simulation_qois = DataCollection("conditioned simulation qois")

        self._weighted_conditioned_experiment_qois = (
            DataCollection("weighted conditioned experiment qois"))
        self._weighted_conditioned_simulation_qois = (
            DataCollection("weighted conditioned simulation qois"))

    def add_weighted_conditioned_experiment_qois(self, qois):
        self._add_data_to_data_collection(self._weighted_conditioned_experiment_qois,
                                          qois, qois.field_names)

    def add_weighted_conditioned_simulation_qois(self, qois):
        self._add_data_to_data_collection(self._weighted_conditioned_simulation_qois, 
                                          qois, qois.field_names)
  
    def get_flattened_weighted_conditioned_experiment_qois(self):
        return self.flatten_data_collection(self._weighted_conditioned_experiment_qois)

    def get_flattened_weighted_conditioned_simulation_qois(self):
        return self.flatten_data_collection(self._weighted_conditioned_simulation_qois)
    
    def set_conditioned_experiment_data(self, data):
        if not self._large_data_sets:
            self._conditioned_experiment_data = data

    def set_conditioned_simulation_data(self, data):
        if not self._large_data_sets:
            self._conditioned_simulation_data = data
        
    def set_experiment_data(self, data):
        if not self._large_data_sets:
            self._experiment_data = data

    def set_simulation_data(self, data):
        if not self._large_data_sets:
            self._simulation_data = data

    def set_conditioned_experiment_qois(self, qois):
        self._conditioned_experiment_qois = qois

    def set_conditioned_simulation_qois(self, qois):
        self._conditioned_simulation_qois = qois
        
    def set_experiment_qois(self, qois):
        self._experiment_qois = qois

    def set_simulation_qois(self, qois):
        self._simulation_qois = qois

    @property
    def simulation_qois(self):
        """
        :return: Returns a DataCollection with the simulation 
            QoIs that were extracted from the 
            simulation data for each state that was evaluated 
            for the objective. The QoIs may be 
            extracted using a MatCal default QoI extractor from 
            module :mod:`~matcal.core.qoi_extractor` 
            or may be a user specified QoI extractor.

        :rtype: :class:`~matcal.core.data.DataCollection`
        """
        return self._simulation_qois

    @property
    def experiment_qois(self):
        """
        :return: Returns a DataCollection with the experimental QoIs 
            that were extracted from the 
            experimental data for each state that was evaluated 
            for the objective. The QoIs may be 
            extracted using a MatCal default QoI extractor from 
            module :mod:`~matcal.core.qoi_extractor` 
            or may be a user specified QoI extractor.

        :rtype: :class:`~matcal.core.data.DataCollection`
        """
        return self._experiment_qois

    @property
    def conditioned_simulation_qois(self):
        """
        :return: Returns a DataCollection with the conditioned simulation 
            QoIs for each state that was 
            evaluated for the objective. Conditioning is done 
            automatically by MatCal 
            in an attempt to normalize all provided experimental data to 
            be on the order of 0-1. The same conditioning
            that is derived from and then applied to the experimental 
            data is applied to the corresponding 
            simulation data by state. See :ref:`MatCal Objective Calculations`.

        :rtype: :class:`~matcal.core.data.DataCollection`
        """
        return self._conditioned_simulation_qois

    @property
    def conditioned_experiment_qois(self):
        """
        :return: Returns a DataCollection with the conditioned experiment 
            QoIs for each state that was 
            evaluated for the objective. Conditioning is done automatically 
            by MatCal 
            in an attempt to normalize all provided experimental data to 
            be on the order of 0-1. The same conditioning
            that is derived from and then applied to the experimental data 
            is applied to the corresponding 
            simulation data by state. See :ref:`MatCal Objective Calculations`.

        :rtype: :class:`~matcal.core.data.DataCollection`
        """
        return self._conditioned_experiment_qois

    @property
    def weighted_conditioned_simulation_qois(self):
        """
        :return: Returns a DataCollection with the weighted and conditioned
            simulation QoIs stored for each 
            state that was evaluated for the objective. Conditioning is 
            done automatically by MatCal 
            in an attempt to normalize all provided 
            experimental data to be on the order of 0-1 and 
            the weighting is what was specified by MatCal 
            defaults and the user. See :ref:`MatCal Objective Calculations`.

        :rtype: :class:`~matcal.core.data.DataCollection`
        """
        return self._weighted_conditioned_simulation_qois

    @property
    def weighted_conditioned_experiment_qois(self):
        """
        :return: Returns a DataCollection with the weighted and 
        conditioned experiment QoIs stored for each 
            state that was evaluated for the objective. 
            Conditioning is done automatically by MatCal 
            in an attempt to normalize all provided experimental 
            data to be on the order of 0-1 and 
            the weighting is what was specified by MatCal defaults 
            and the user. See :ref:`MatCal Objective Calculations`.

        :rtype: :class:`~matcal.core.data.DataCollection`
        """
        return self._weighted_conditioned_experiment_qois

    @property
    def conditioned_simulation_data(self):
        """
        :return: Returns a DataCollection with the conditioned 
            simulation data for each state that was 
            evaluated for the objective. Conditioning is 
            done automatically by MatCal 
            in an attempt to normalize all provided experimental 
            data to be on the order of 0-1. The same conditioning
            that is derived from and then applied to the 
            experimental data is applied to the corresponding 
            simulation data by state. See :ref:`MatCal Objective Calculations`.

        :rtype: :class:`~matcal.core.data.DataCollection`
        """
        return self._conditioned_simulation_data

    @property
    def conditioned_experiment_data(self):
        """
        :return: Returns a DataCollection with the conditioned 
            experimental data for each state that was 
            evaluated for the objective. Conditioning 
            is done automatically by MatCal 
            in an attempt to normalize all provided experimental 
            data to be on the order of 0-1. The same conditioning
            that is derived from and then applied to 
            the experimental data is applied to the corresponding 
            simulation data by state. See :ref:`MatCal Objective Calculations`.

        :rtype: :class:`~matcal.core.data.DataCollection`
        """

        return self._conditioned_experiment_data

    @property
    def simulation_data(self):
        """
        :return: Returns a DataCollection with the simulation data generated by the 
            appropriate model for the objective.
            
        :rtype: :class:`~matcal.core.data.DataCollection`
        """
        return self._simulation_data

    @property
    def experiment_data(self):
        """
        :return: Returns a DataCollection with the experimental data supplied by the user 
            for the objective.

        :rtype: :class:`~matcal.core.data.DataCollection`
        """
        return self._experiment_data
    
    
class ObjectiveResults(ObjectiveContainerBase):
    """
    The ObjectiveResults objects stores all relevant data needed to calculate the objective 
    value for a user requested objective.
    """

    def __init__(self, required_fields, fields_of_interest, large_data_sets=False):
        super().__init__(required_fields, fields_of_interest, large_data_sets)

        self._residuals = None
        self._conditioned_residuals = None
        self._weighted_residuals = None
        self._weighted_conditioned_residuals = DataCollection("weighted conditioned residuals")
        self._weighted_conditioned_normalized_residuals = None
                
        self._weighted_conditioned_objectives = DataCollection("weighted conditioned objectives")
        self._objective = None

        self._initialize_nonessential_qois()

    def _initialize_nonessential_qois(self):
        self._residuals = DataCollection("residuals")
        self._conditioned_residuals = DataCollection("conditioned residuals")
        self._weighted_residuals = DataCollection("weighted_residuals")

    def set_objective(self,  objective_value):
        self._objective = objective_value

    def set_weighted_conditioned_normalized_residuals(self, new_residuals):
        if not isinstance(new_residuals, DataCollection):
            raise self.InvalidResidualError(new_residuals)
        self._weighted_conditioned_normalized_residuals = new_residuals

    def add_residuals(self, residual_vector):
        self._add_data_to_data_collection(self._residuals, residual_vector, 
                                          self._fields_of_interest)

    def add_conditioned_residuals(self, residual_vector):
        self._add_data_to_data_collection(self._conditioned_residuals, 
                                          residual_vector, self._fields_of_interest)

    def add_weighted_residuals(self, residual_vector):
        self._add_data_to_data_collection(self._weighted_residuals, 
                                          residual_vector, self._fields_of_interest)

    def add_weighted_conditioned_residuals(self, weighted_residual_vector):
        self._add_data_to_data_collection(self._weighted_conditioned_residuals, 
                                          weighted_residual_vector, self._fields_of_interest)

    def add_weighted_conditioned_objective(self, objective_value):
        self._add_data_to_data_collection(self._weighted_conditioned_objectives, 
                                          objective_value, self._fields_of_interest)
   
    def get_flattened_weighted_conditioned_residuals(self):
        return self.flatten_data_collection(self._weighted_conditioned_residuals)

    def get_flattened_weighted_conditioned_objectives(self):
        return self.flatten_data_collection(self._weighted_conditioned_objectives)

    def get_objective(self):
        return self._objective

    @property
    def objectives(self):
        """
        :return: Returns a DataCollection with objective values stored for each 
            state that was evaluated for the objective.

        :rtype: :class:`~matcal.core.data.DataCollection`
        """
        return self._weighted_conditioned_objectives

    @property
    def residuals(self):
        """
        :return: Returns a DataCollection with the residual values stored for each 
            state that was evaluated for the objective.

        :rtype: :class:`~matcal.core.data.DataCollection`
        """
        return self._residuals

    @property
    def conditioned_residuals(self):
        """
        :return: Returns a DataCollection with the 
            conditioned residual values stored for each 
            state that was evaluated for the objective. 
            Conditioning is done automatically by MatCal 
            to normalize all provided experimental data to 
            be on the order of 0-1. See :ref:`Conditioning`.

        :rtype: :class:`~matcal.core.data.DataCollection`
        """
        return self._conditioned_residuals

    @property
    def weighted_residuals(self):
        """
        :return: Returns a DataCollection with the weighted residual values stored for each 
            state that was evaluated for the objective.

        :rtype: :class:`~matcal.core.data.DataCollection`
        """
        return self._weighted_residuals

    @property
    def weighted_conditioned_residuals(self):
        """
        :return: Returns a DataCollection with the weighted and 
            conditioned residual values stored for each 
            state that was evaluated for the objective. Conditioning 
            is done automatically by MatCal 
            in an attempt to normalize all provided experimental data 
            to be on the order of 0-1 and 
            the weighting is what was specified by MatCal defaults 
            and the user. See :ref:`MatCal Objective Calculations`.

        :rtype: :class:`~matcal.core.data.DataCollection`
        """
        return self._weighted_conditioned_residuals

    @property
    def weighted_conditioned_normalized_residuals(self):
        """
        :return: Returns a DataCollection with the weighted, 
            conditioned and normalized residual values stored for each 
            state that was evaluated for the objective. Conditioning 
            is done automatically by MatCal 
            in an attempt to normalize all provided experimental data 
            to be on the order of 0-1 and 
            the weighting is what was specified by MatCal defaults 
            and the user. See :ref:`MatCal Objective Calculations`.

        :rtype: :class:`~matcal.core.data.DataCollection`
        """
        return self._weighted_conditioned_normalized_residuals


    @property
    def calibration_residuals(self):
        """
        :return: Returns an NumPy array that is the concatenated, 
            weighted, conditioned, and normalized
            residual of this objective. This is the residual that 
            is used by MatCal for calibration. 
        """
        if self._weighted_conditioned_normalized_residuals is None:
            return None
        else:
            return self.flatten_data_collection(self._weighted_conditioned_normalized_residuals)
