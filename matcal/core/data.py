"""
The data module contains classes and functions for converting 
data into the structure that MatCal requires for studies.
"""

from matcal.core.utilities import _time_interpolate
from matcal.core.serializer_wrapper import _format_serial
import numpy as np
from itertools import count
from collections import OrderedDict
import numbers
from abc import ABC, abstractmethod
import os

from matcal.core.state import SolitaryState, State, StateCollection
from matcal.core.utilities import (ContainerCollectionBase, check_value_is_real_between_values, 
                                   check_value_is_positive_integer, check_value_is_bool)

from matcal.core.logger import initialize_matcal_logger
logger = initialize_matcal_logger(__name__)


class Data(np.ndarray):
    """
    Data is the base data structure for all MatCal data. This data structure is
    an interface to data that are
    used for MatCal studies. It is derived from a NumPy ndarrays 
    but adds name and state, so that the data can be
    uniquely identified.

    Accessing fields through field names returns the data for that field in 
    either 1D or 2D arrays. If the data is 'global' such as time or load, 
    the data will be reported as a 1D [n_times] array. If the data is field
    based the data is reported back as a 2D 
    [n_times, n_points] array. 
    """
    _id_numbers = count(0)

    class TypeError(RuntimeError):
        def __init__(self, *args):
            super().__init__(*args)

    class KeyError(RuntimeError):
        def __init__(self, *args):
            super().__init__(*args)

    class ValueError(RuntimeError):
        def __init__(self, *args):
            super().__init__(*args)

    def __new__(cls, data, state=SolitaryState(), name=None):
        """
        :param data: data to be added to the MatCal data object.
        :type data: ArrayLike

        :param state: the state associated with the data. If none is passed it 
            will be assigned the default state.
        :type state: :class:`~matcal.core.state.State`

        :param name: the name for the data. By default it is set to "data_set_#" 
            name with a unique id number. If
            :func:`~matcal.core.data_importer.FileData` is used to import data,
            then its name is set to the
            filename from which the data was imported.
        :type name: str
        """
        cls._check_type(cls, data, (np.ndarray, np.record), "data passed to MatCal Data")
        obj = np.asarray(data).view(cls) #view will cast all intermal arrays as cls[Data] as well

        obj._state = None
        obj.set_state(state)
        obj._id_number = next(cls._id_numbers)
        obj._name = "data_set_{}".format(obj._id_number)
        if name is not None:
            obj.set_name(name)

        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self._state = getattr(obj, '_state', None)
        self._id_number = getattr(obj, '_id_number', None)
        self._name = getattr(obj, '_name', None)

    def __array_wrap__(self, out_array, context=None):
        return np.ndarray.__array_wrap__(self, out_array, context)

    def set_state(self, state):
        """
        Sets the optional state value for the data.

        :param state: The state for this particular data set.
        :type state: :class:`~matcal.core.state.State`
        """
        self._check_type(state, State, "state for the data set")
        self._state = state

    def set_name(self, name):
        """
        Sets the optional name value for the data. If the data is imported using
        :func:`~matcal.core.data_importer.FileData`, the name is set to the 
        filename from which the data was
        imported. If no name is passed and the data was created from the 
        constructor or another function,
        an arbitrary name will be given to the data.

        :param name: The name for this particular data set.
        :type name: str
        """
        self._check_type(name, str, "name for the data set")
        self._name = name

    def add_field(self, field_name, data):
        """ 
        Adds a new 1D field to the data and returns the 
        updated data. The original data object is not modified. 
        The added field must have the 
        same length as the existing fields. 

        :param field_name: The name of the new field to be added.
        :type field_name: str

        :param data: the data to be added.
        :type data: ArrayLike

        :return: the data with newly added field
        :rtype: `~matcal.core.data.Data`
        """
        self._check_type(field_name, str, "added field")
        if len(data) != self.length:
            error_str = (f"Field to be added '{field_name}' has length " +
                         f"{len(data)}. It must be of length " +
                         f"{self.length}.")
            raise self.ValueError(error_str)
        data_dict = convert_data_to_dictionary(self)
        data_dict.update({field_name:data})
        updated_data = convert_dictionary_to_data(data_dict)
        updated_data.set_state(self.state)
        return updated_data
    
    def _check_type(self, variable, desired_type, message_name):
        if not isinstance(variable, desired_type):
            raise self.TypeError("The {} must be of type {}. "
                "Received a variable of type {}".format(message_name, desired_type, type(variable)))

    def _check_field_in_data(self, field):
        if field not in self.field_names:
                raise self.KeyError(f"The field \"{field}\" does not exist. "+
                f"The following fields exist in the data:\n{self.field_names}")

    @property
    def length(self):
        """
        :return: The length of the data for each field.
        :rtype: int
        """
        if len(self.shape) > 0:
            return self.shape[0]
        else: 
            return 1

    @property
    def state(self):
        """
        :return: The physical state of the data corresponding to the experimental conditions.
        :rtype: :class:`~matcal.core.state.State`
        """
        return self._state

    @property
    def field_names(self):
        """
        :return: list of strings of all field names.
        :rtype: list
        """
        field_names = self.dtype.names
        if field_names is None:
            return []
        else:
            return list(field_names)

    def keys(self):
        return self.field_names

    @property
    def name(self):
        """
        Returns the name for the data. If the data is imported using
        :func:`~matcal.core.data_importer.FileData`, the name is set to the 
        filename from which the data was
        imported. If no name is passed and the data was created from the 
        constructor or another function,
        an arbitrary name will be given to the data.

        :rtype name: str
        """
        return self._name

    def remove_field(self, field):
        """Returns a copy of the Data class with the desired field removed. The 
        original data object is not modified.
        
        :rtype: :class:`~matcal.core.data.Data`
        """
        self._check_type(field, str, "Data field to be removed")
        self._check_field_in_data(field)
        updated_field_names = self.field_names
        updated_field_names.remove(field)
        if len(updated_field_names) == 0:
            return Data(np.array([]), self.state, self.name)
        else:
            return self[updated_field_names].copy()

    def rename_field(self, old_name, new_name):
        """
        Returns the Data class with the desired the field name changed. 
        Note that the old name is overwritten and not
        saved.

        :param old_name: the old field name that is to be updated
        :type old_name: str

        :param new_name: the replacement field name for the field name that is being changed.
        :type new_name: str
        """
        self._check_type(old_name, str, "old field name")
        self._check_type(new_name, str, "new field name")
        self._check_field_in_data(old_name)
        old_field_names = self.field_names
        old_name_index = old_field_names.index(old_name)
        new_field_names = old_field_names
        new_field_names[old_name_index] = new_name
        self.dtype.names = new_field_names
        return self

    def __eq__(self, value) -> bool:
        return super().__eq__(value)
    
    def __reduce__(self):
        # Get the parent's __reduce__ tuple
        pickled_state = super(Data, self).__reduce__()
        # Create our own tuple to pass to __setstate__, 
        # but append the __dict__ rather than individual members.
        new_state = pickled_state[2] + (self.__dict__,)
        # Return a tuple that replaces the parent's __setstate__ tuple with our own
        return (pickled_state[0], pickled_state[1], new_state)

    def __setstate__(self, state):
        self.__dict__.update(state[-1])  # Update the internal dict from state
        # Call the parent's __setstate__ with the other tuple elements.
        super(Data, self).__setstate__(state[0:-1])


class DataCollection(ContainerCollectionBase):
    """
    A collection of :class:`~matcal.core.data.Data` objects to be used for a study. No
    restrictions are enforced on the type or contact of :class:`~matcal.core.data.Data` objects
    added to the collection. However, they are meant to hold data that is related by experiment
    and should generally have the same if not similar fields. 
    
    Exceptions to this rule may be 
    when two different types of data are taken from the same experiment using different 
    data acquisition hardware. In this case it may make sense to store 
    :class:`~matcal.core.data.Data` objects
    in a data collection with different fields.

    .. warning::
        Not all MatCal objects or methods support data collections with 
        :class:`~matcal.core.data.Data`
        objects that contain different field names. Appropriate errors should be 
        used if such data collections
        are passed to them.
    """
    _collection_type = Data

    def __init__(self, name, *data_sets):
        """
    :param name: The name of this data collection.
    :type name: str

    :param data_sets: The :class:`~matcal.core.data.Data` sets to be added to the collection.
    :type data_sets: list(:class:`~matcal.core.data.Data`) or
          :class:`~matcal.core.data.Data`.

    :raises CollectionValueError: If name is a an empty string.
    :raises CollectionTypeError: If name is not a string and the data objects 
        to be added to the collection are
        not of the correct type.
    """
        self._field_names = []
        super().__init__(name, *data_sets)

    @property
    def field_names(self):
        """
        :return: a list of field names that exist in the data collection. 
            These may not exist in all data objects or states and may only be
            in one data object in the collection.
        """
        self._field_names = []
        for data_list in self._items.values():
            for data in data_list:
                for field_name in data.field_names:
                    if field_name not in self._field_names:
                        self._field_names.append(field_name)
        return self._field_names

    @property
    def state_names(self):
        """
    :return: the names of the :class:`~matcal.core.state.State` objects in the data collection.
    :rtype: list(str)
    """
        state_names = []
        for state, item in self._items.items():
            state_names.append(state.name)
        return state_names

    @property
    def states(self):
        """
    :return: The state :class:`~matcal.core.state.State` objects in the data collection.
    :rtype: :class:`~matcal.core.state.StateCollection`
    """
        sc = StateCollection('data states')

        for key, item in self._items.items():
            sc.add(key)

        return sc

    def state_field_names(self, state):
        """
        Return all the field names in all Data objects for the given state. 
        Note that not all Data objects need to have all field names. This 
        is just a comprehensive list of field names that exist across all Data 
        objects in the DataCollection for this state.
        
        :param state: the state of interest to get all field names for
        :type state: str or :class:`~matcal.core.state.State`

        :return: a list of all field names
        :rtype: list(str)
        """
        state_field_names = []
        for data in self.__getitem__(state):
            state_field_names += data.field_names
        
        return list(set(state_field_names))

    def state_common_field_names(self, state):
        """
        Return all the field names common to all Data objects for the given state. 
        
        :param state: the state of interest to get all field names for
        :type state: str or :class:`~matcal.core.state.State`

        :return: a list of all field names that are common to all data sets for that state
        :rtype: list(str)
        """
        from copy import deepcopy
        state_field_names = self.state_field_names(state)
        common_state_field_names = deepcopy(state_field_names)

        for state_field_name in state_field_names:
            for data in self.__getitem__(state):
                if state_field_name not in data.field_names:
                    if state_field_name in common_state_field_names:
                        common_state_field_names.remove(state_field_name)
        return common_state_field_names

    def add(self, item):
        """
        Add a :class:`~matcal.core.data.Data` object to a data collection.

        :param item: Data object to be added to the data collection.
        :type item: :class:`~matcal.core.data.Data`
        """
        if isinstance(item, list):
            for it in item: self.add(it)
            return
        super()._check_item_is_correct_type(item)
        self._add_data(item)
        self._add_new_field_names(item)

    def remove_field(self, field_name):
        """
        Removes the field from all data sets stored
        in the data collection that have the passed field name. 
        If the data collection does not have any data sets with the
        specified field name, a warning will be sent to MatCal output.

        :param field_name: the name of the field to remove
        :type field_name: str
        """
        valid_field = self._valid_field_name(field_name)
        if valid_field:
            for state, data_list in self._items.items():
                for index, data in enumerate(data_list):
                    if field_name in data.field_names:
                        self._items[state][index] = data.remove_field(field_name)
        else:
            logger.warning(f"The field \"{field_name}\" is not in DataCollection "+
                           f"\"{self.name}\" and will not be removed")
    
    def _valid_field_name(self, field_name):
        if not isinstance(field_name, str):
            raise self.CollectionTypeError(f"The field passed to the DataCollection "
            f"\"remove_field\" method must be a string. Received \"{field_name}\".")

        return field_name in self.field_names

    def _add_data(self, data):
        if data.state.name in self.state_names and data.state not in self.states.values():
            raise self.NonUniqueStateNameError(self.state_names, data.state, self.states)
        super().add(data.state, data)

    def _add_new_field_names(self, data):
        for field_name in data.field_names:
            if field_name not in self._field_names:
                self._field_names.append(field_name)

    def __getitem__(self, key):
        if isinstance(key, str):
            if key in self.state_names:
                key = self.states[key]
            else:
                err_msg = f"State named \"{key}\" not in the DataCollection \"{self.name}\".\n"
                err_msg += f"Available states are: {list(self.states.keys())}"
                raise KeyError(err_msg)
        if not isinstance(key, State):
            err_msg = (f"Getting items from DataCollection requires a state or "
                        f"state name as a key. Passed a variable of type \"{type(key)}\', "
                        f"value \"{key}\".")
            raise KeyError(err_msg)
        return self._items[key]

    class NonUniqueStateNameError(RuntimeError):
        
        def __init__(self, names, new_state, states):
            message = ('Attempting to add a different data state with '+
                       'the same name as a different data state:')
            message += f'\nExisting names: {names}'
            message += f"\n New state: {new_state}"
            message += f"\n Existing states: {states}"
            super().__init__(message)

    def dumps(self, ignore_point_data=False):
        dump_data = {}
        for state in self.states:
            state_data = self.__getitem__(state)
            dump_data[state] = []
            for sd in state_data:
                processed_data = dict(convert_data_to_dictionary(sd))
                p_data = {}
                for name, value in processed_data.items():
                    converted_data = np.atleast_1d(value.astype(float))
                    if ignore_point_data and converted_data.ndim>1 and converted_data.shape[1]>1:
                        continue
                    p_data[name] = converted_data.tolist()
                dump_data[state].append(p_data)
        return dump_data

    def plot(self, independent_field: str, dependent_field: str, plot_function=None, 
            figure=None, show: bool=True, labels: str=None, state: State=None,
            block: bool=True, **kwargs) -> None:
        """
        Plots the data with the independent field on the horizontal axis and 
        dependent field on the vertical axis. It plots each state on a separate figure.

        :param independent_field: field name to use as horizontal axis variable.
        :type independent_field: str

        :param dependent_field: field name to use as vertical axis variable.
        :type dependent_field: str

        :param plot_function: a valid matplotlib plot function such as plot, semilogx, etc
        :type plot_function: matplotlib plot function

        :param figure: a valid matplotlib Figure for the data collection to be plotted on.
        :type figure: matplotlib Figure

        :param show: option to show or not show plot
        :type show: bool

        :param labels: provide a label for each data set other than the data set name. 
            This can take the form of "suppress", 
            "{user_provided_label}" or "{user_provided_label} (#)".  
            If "suppress" is passed, none of the data
            will be labeled. If "{user_provided_label}" is passed,  
            the first data set will be labeled once
            as "{user_provided_label}" where "user_provided_label" can be any
            user provided string. The rest will not be 
            labeled. If the 
            last option is used,  where labels="{user_provided_label} (#)", each data set will 
            be labeled with "{user_provided label}" and a number based on the 
            order it is pulled from the data set.
            For example, a data collection with three data sets and this 
            function called with labels="experiment (#)", the labels
            will be "experiment 1", "experiment 2", "experiment 3".
        :type labels: str

        :param state: specify a specific state to plot using the state name or state object
        :type state: :class:`~matcal.core.state.State` or str

        :param block: stops Python from executing code after the plot figure is created. Follow-on code
            will not execute until the figure is closed.
            Default is to block (e.g. block=True).
        :type block: bool
 
        :param kwargs: a set of valid keyword argument pairs for the Matplotlib plotting function
        :type kwargs: dict(str, str)
        """
        import matplotlib.pyplot as plt

        user_state = state
        user_figure = figure
        
        if plot_function is None:
            plot_function = plt.plot

        label_count = 0 
        if user_state is not None:
            self._plot_state_data_list(self[user_state], plt, plot_function, dependent_field, 
                independent_field, user_figure, labels, label_count, kwargs)
        else:
            for state in self.keys():
                self._plot_state_data_list(self[state], plt, plot_function, dependent_field, 
                    independent_field, user_figure, labels, label_count, kwargs)
           
        if show:
            plt.show(block=block)    

    def _plot_state_data_list(self, data_state_list, plt, plot_function, 
            dependent_field, independent_field, user_figure, labels, label_count, kwargs):
        state =data_state_list[0].state
        self._set_figure(user_figure, state, independent_field, dependent_field)
        if "linestyle" not in kwargs.keys() and "marker" not in kwargs.keys():
            kwargs["linestyle"] = '' 
            kwargs["marker"] = '.'
        for data_index, data in enumerate(data_state_list):
            if independent_field in data.field_names and dependent_field in data.field_names:
                label = self._get_plot_label(labels, data.name, data_index, label_count)
                label_count +=1
                plot_function(data[independent_field], data[dependent_field], label=label, **kwargs)
            else:
                logger.warning(f"Skipping plotting for data \"{data.name}\" in DataCollection "+
                    f"\"{self.name}\". The independent and dependent " 
                    "fields are not in the data.") 
        plt.xlabel(independent_field)
        plt.ylabel(dependent_field)
        if not user_figure:
            plt.title(state.name)
        plt.legend()
        
    def _set_figure(self, figure, state: State, independent_field: str, dependent_field: str):
        import matplotlib.pyplot as plt

        valid_user_fig_provided = self._check_valid_user_fig_provided_for_plot(figure)
        if not valid_user_fig_provided:
            figure = plt.figure(state.name+" "+independent_field+" "+dependent_field, 
                                constrained_layout=True)
        else:
            plt.figure(figure.number, constrained_layout=True)
        return figure

    def _check_valid_user_fig_provided_for_plot(self, figure):
        from matplotlib.figure import Figure

        valid_user_fig_provided = False
        if figure is not None and isinstance(figure, Figure):
            valid_user_fig_provided = True
        elif figure is not None:
            raise self.CollectionTypeError("Invalid figure passed to DataCollection.plot(). "
                f"Received type \"{type(figure)}\", but expected a matplotlib Figure.")

        return valid_user_fig_provided

    def _get_plot_label(self, labels, data_name, index, label_count):
        if labels is None:
            return self._get_default_label(data_name)
        default_label = self._get_default_label(data_name)
        if labels=="suppress":
            return "_"+default_label
        elif "(#)" in labels:
            return labels.replace("(#)", str(index))
        elif labels is not None:
            if label_count > 0:
                labels = "_"+labels
            return labels
        
    def _get_default_label(self, data_name):
        split_data_name = os.path.split(data_name)
        default_label = os.path.split(data_name)[-1]
        return default_label

    def get_data_by_state_values(self, **kwargs):
        """
        Get a :class:`~matcal.core.data.DataCollection` containing
        data that has the state variables with values passed into 
        this method. 

        :param kwargs: keyword/value pairs of the desired state variables
        :type kwargs: dict(str, str or float)

        :return: all data in the data collection that have states with the 
            state variable and values specified in kwargs.
        :rtype: :class:`~matcal.core.data.DataCollection`
        """
        data_col_with_state_vals = DataCollection(self._get_sub_selection_name(kwargs))
        all_data_has_state_vals = True
        for state in self.keys():        
            values_in_state = self._dict_in_state_params(kwargs, state)
            if values_in_state:
                data_col_with_state_vals.add(self[state])
            all_data_has_state_vals = (values_in_state and 
                                       all_data_has_state_vals)
        if all_data_has_state_vals:
            data_col_with_state_vals = self
        return data_col_with_state_vals

    def _get_sub_selection_name(self, kwargs):
        new_name = self.name+"_with_state_params"
        for key, val in kwargs.items():
                new_name += f"_{key}_{val}"
        return new_name
    
    def _dict_in_state_params(self, dictionary, state):
        for key, val in dictionary.items():
            if key not in state.params:
                return False
            else:
                if state.params[key] != val:
                    return False
        return True

    def get_states_by_state_values(self, **kwargs):
        """
        Get a :class:`~matcal.core.state.StateCollection` containing
        the states with the state variable values passed into 
        this method. 

        :param kwargs: keyword/value pairs of the desired state variables
        :type kwargs: dict(str, str or float)

        :return: a state collection that has all states with the 
            state variables and values specified in kwargs.
        :rtype: :class:`~matcal.core.state.StateCollection`
        """
        return self.get_data_by_state_values(**kwargs).states

    def report_statistics(self, independent_field:str) -> dict:
        """
        Get a summary of the statistics information. The method will report the 
        mean and standard deviation for all dependent fields across the independent
        within each state. The data will be collocated to a common set of locations
        within the independent field. Statistics near the limits of the independent
        field range may be less accurate than of those in the interior because of 
        errors due to extrapolation that may occur in the collocation process. 
        
        :param independent_field: The string to designate which field should be 
            interpreted as the independent field. 
        :type independent_field: str
        
        :return: a dictionary that contains the statistical measurements of the data fields.
            the data is organized by [field_name][state_name][stat_name]
        :rtype: dict
        """
        stats_tool = DataCollectionStatistics()
        stats_report = {}
        for state, state_data in self._items.items():
            stats_report[state.name] = stats_tool.generate_state_statistics(independent_field,
                                                                            self, state)
        return stats_report
        

class DataCollectionStatistics:

    def __init__(self, num_interpolation_points=None, sort_ascending=True, interpolation_tool=None, 
                 **interp_keyword_arguments):
        """
        This class can be used to calculate basic statistics on the data in a data collection
        by state and field. By default it calculates the mean and standard deviation of the data.
        It can also be used to calculate the percentiles at user specified percentile values. 
        This class assumes the data is repeated 1D data with an independent field. It will 
        interpolate the data to a common set of independent field values and then 
        calculate the statistics at each of these values. For the independent field, 
        the maximum and minimum values will be the maximum and minimum values 
        for that field from all repeats for the state of interest.

        :param num_interpolation_points: Select the number of independent fields to interpolate the 
            dependent fields data to. By default this sets the number of points to the
            average length of all repeat data for the specified state and field. 
        :type num_interpolation_points: int
        
        :param sort_ascending: sort the data according to the independent variable 
            before interpolating.
        :type sort_ascending: bool

        :param interpolation_tool: The data for a given state and independent field are interpolated 
            to a common set of independent field values. Optionally select the interpolation method 
            used with this parameter. The interpolation method by default is NumPy.interp.
            To change the interpolation method, pass in an appropriate SciPy 1D 
            interpolation class or function such as `make_interp_spline` or other 
            similar interpolation tool that builds a callable interpolation object 
            that takes the independent values and dependent values with optional 
            keyword arguments on initialize. The callable object created will return 
            interpolation values for passed independent variable values
        :type interpolation_tool: func or class
        
        :interp_keyword_arguments: optional keyword arguments that are valid 
            for the given interpolation tool. If an interpolation tool 
            is not passed, these must be valid keyword arguments for NumPy.interp.
        :interp_keyword_arguments: dict(str,(float,str))
        """    
        self._analysis_to_perform = OrderedDict()
        self._analysis_to_perform['mean'] = _mean
        self._analysis_to_perform['std dev'] = _std_dev
        self._num_interpolation_points = None
        self.set_number_of_interpolation_points(num_interpolation_points)
        self._interpolation_tool = None
        self._interpolation_kwargs = {}
        self.set_interpolation_tool(interpolation_tool, **interp_keyword_arguments)
        self._percentiles = []
        self._sort_ascending = False
        self.set_sort_ascending(sort_ascending)

    def set_interpolation_tool(self, interpolation_tool=None, **interp_keyword_arguments):
        """
        Change the interpolation tool and associated keyword arguments.

        :param interpolation_tool: The data for a given state and independent field are interpolated 
            to a common set of independent field values. Optionally select the interpolation method 
            used with this parameter. The interpolation method by default is NumPy.interp.
            To change the interpolation method, pass in an appropriate SciPy 1D 
            interpolation class or function such as `make_interp_spline` or other 
            similar interpolation tool that builds a callable interpolation object 
            that takes the independent values and dependent values with optional 
            keyword arguments on initialize. The callable object created will return 
            interpolation values for passed independent variable values
        :type interpolation_tool: func or class
        
        :interp_keyword_arguments: optional keyword arguments that are valid 
            for the given interpolation tool. If an interpolation tool 
            is not passed, these must be valid keyword arguments for NumPy.interp.
        :interp_keyword_arguments: dict(str,(float,str))
        """
        self._interpolation_tool=interpolation_tool
        self._interpolation_kwargs = interp_keyword_arguments

    def set_sort_ascending(self, sort_ascending=True):
        """
        Automatically sort the data so that the independent variable is ascending. 
        This is necessary for some interpolation methods to get a valid interpolation.

        :param sort_ascending: Flag to turn sorting off/on.
        :type sort_ascending: bool
        """
        check_value_is_bool(sort_ascending, "sort_ascending", 
                            "DataColleciontStatistics")
        self._sort_ascending = sort_ascending

    def set_number_of_interpolation_points(self, num_interpolation_points):
        """
        Manually select the number of points for the interpolation of 
        the dependent fields.

        :param num_interpolation_points: the number of points for interpolation
        :type num_interpolation_points: int
        """
        if num_interpolation_points is not None:
            check_value_is_positive_integer(num_interpolation_points, "num_interpolation_points", 
                                            "DataCollectionStatistics")
            self._num_interpolation_points = num_interpolation_points
        
    def set_percentiles_to_evaluate(self, *percentiles):
        """
        Set the percentiles to evaluate for the data sets of interest. 
        Calling this will remove any preexisting percentiles 
        previously requested.

        :param percentiles: Specify percentiles of interest for the data set. 
        :type percentiles: list(float)
        """
        self._percentiles = []
        for percentile in percentiles:
            check_value_is_real_between_values(percentile, 0,100, 
                "percentiles", "DataCollectionStatistics.set_percentiles_to_evaluate", 
                closed=True)
            self._percentiles.append(percentile)

    def generate_state_statistics(self, indep_field, data_collection, state):
        """
        Calculate the requested statistics on the DataCollection of 
        interest for the given state and independent field. 

        :param indep_field: the desired independent field for interpolation and 
            subsequent statistics calculation.
        :type indep_field: str 

        :param data_collection: the data collection that includes the data 
            for the statistics calculations.
        :type data_collection::class:`~matcal.core.data.DataCollection`

        :param state: the state of interest for the current calculation
        :type state: :class:`~matcal.core.state.State`     

        :return: A nested dictionary of the statistics results. The first key is the 
            name of the fields for which the statistics were evaluated. The second key is the 
            statistic that was calculated. These include "mean", "std dev", and "percentile_#".
        :rtype: dict(str, Array-Like[float])    
        """
        self._verify_generate_stats_inputs(data_collection, indep_field, state)
        interped_data = self._interpolate_state_data_to_common_independent_variable(indep_field, 
            data_collection, state)
        state_report = {f"locations": interped_data.pop(indep_field)}
        for field in interped_data:
            field_data = np.array(interped_data[field])
            field_report = {}
            for stat_name, stat_fun in self._analysis_to_perform.items():
                field_report[stat_name] = stat_fun(field_data)
            if self._percentiles:
                for percentile in self._percentiles:
                    field_report[f"percentile_{percentile}"] = np.percentile(field_data, percentile,
                        axis=0)
            state_report[field] = field_report
        
        return state_report
    
    def _verify_generate_stats_inputs(self, data_collection, indep_field, state):
        if state not in data_collection:
            raise KeyError(f"The state \"{state.name}\" is not in "+
                           f"data collection \"{data_collection.name}\".")

        for idx, data in enumerate(data_collection[state]):
            if indep_field not in data.field_names:
                raise KeyError(f"The independent field \"{indep_field}\" is not in "+
                               f"data set {idx} in the data collection \"{data_collection.name}\".")

    def _interpolate_state_data_to_common_independent_variable(self, indep_field, 
        data_collection, state, indep_field_data_collection=None):
        if indep_field_data_collection is None:
            indep_field_data_collection = data_collection
        state_data = data_collection[state]
        field_names = data_collection.state_field_names(state)
        interp_locations = self._make_interpolation_domain(indep_field,
            indep_field_data_collection, state)
        interpolated_data = self._generate_interpolated_data_per_state_by_field(indep_field, 
            state_data, field_names, interp_locations, indep_field_data_collection[state])
        return interpolated_data

    def _generate_interpolated_data_per_state_by_field(self, indep_field, 
        state_data,field_names,interp_locations, indep_field_state_data):
        interpolated_data_by_field = {}        
        for field in field_names:
            interpolated_data_by_field[field] = []
        interpolated_data_by_field[indep_field] = interp_locations
        for cur_data, indep_field_cur_data in zip(state_data, indep_field_state_data):
            for field in field_names:
                if field in cur_data.field_names and field != indep_field:
                    try:
                        interped_data = self._interpolate(interp_locations, 
                            indep_field_cur_data[indep_field], cur_data[field])
                        interpolated_data_by_field[field].append(interped_data)
                    except Exception as e:
                        self._raise_stats_error(indep_field, field, cur_data, e)
        return interpolated_data_by_field

    def _interpolate(self, interp_locs, independent_data, dependent_data):
        if self._sort_ascending:
            sorted_indices = np.argsort(independent_data)
            independent_data = independent_data[sorted_indices]
            dependent_data  = dependent_data[sorted_indices]
        if self._interpolation_tool is None:
            return np.interp(interp_locs, independent_data, dependent_data, 
                **self._interpolation_kwargs)
        else:
            interpolator = self._interpolation_tool(independent_data, dependent_data, 
                **self._interpolation_kwargs)
            return interpolator(interp_locs)

    def _raise_stats_error(self, indep_field, f_name, cur_data, exception):
        message =  "Error Generating Stats for:\n"
        message += f"Field: {f_name}\n"
        message += f"Indep Data: {cur_data[indep_field]}\n"
        message += f"Field Data: {cur_data[f_name]}\n"
        message += f"{repr(exception)}"
        raise RuntimeError(message)

    def _make_interpolation_domain(self, indep_field, data_collection, state):
        state_data = data_collection[state]
        n_points = self._get_number_of_field_points(indep_field, state_data)
        state_max = None
        state_min = None
        for data in state_data:
            x = data[indep_field]
            state_max = self._update_term(state_max, x, np.max)
            state_min = self._update_term(state_min, x, np.min)
        interp_locations = np.linspace(state_min, state_max, n_points)
        return interp_locations

    def _get_number_of_field_points(self, indep_field, state_data):
        if self._num_interpolation_points is None:
            n_points_list = []
            for data in state_data:
                n_points_list.append(len(data[indep_field]))
            n_points = int(np.ceil(np.average(n_points_list)))
            return n_points
        else:
            return self._num_interpolation_points

    def _update_term(self, term, x, f):
        if term is None:
            r_val = f(x)
        else:
            r_val = f([f(x), term])
        return r_val


def _std_dev(data_array):
    return np.std(data_array, axis=0)


def _mean(data_array):
    return np.mean(data_array, axis=0)


class Scaling(object):
    """
    This class is used to apply a scaling multiplier and 
    an offset to a specific field of a :class:`~matcal.core.data.Data` class.
    The offset is applied first, followed by the scale factor.
    """
    class ScalingTypeError(Exception):
        def __init__(self, *args):
            super().__init__(*args)

    def __init__(self, field, scalar=1, offset = 0):
        """
        :param field: The name of the field to be scaled.
        :type field: str

        :param scalar: The magnitude of the scaling to be applied to the specified field.
        :type scalar: float

        :param offset: The magnitude of the offset to be applied to the specified field.
        :type scalar: float


        :raises TypeError: If the scaling object name and the field names are not strings.
        :raises TypeError: If the scalar value passed in is not a number.
        """
        if not isinstance(field, str):
            raise self.ScalingTypeError(f"The field to be scaled must be of type str. \'{field}\' "+
                                        "was passed as the field which is of "+
                                        f"type \'{type(field)}\'.")

        self._field = field
        self._scalar = None
        self._offset = offset
        self.set_scalar(scalar)

    @property
    def field(self):
        """
    :return: The name of the field to be scaled by the scaling object.
    :rtype: str
    """
        return self._field

    def apply_to_data(self, data):
        """
    :param data: the data object with the desired field to be scaled.
    :type data: :class:`~matcal.core.data.Data`

    :return: The data object with the appropriately scaled field
    :rtype: :class:`~matcal.core.data.Data`
    """
        scaled_data = data.copy()
        scaled_field_data = self._scalar * (scaled_data[self._field] + self._offset)
        scaled_data[self.field] = scaled_field_data

        return scaled_data

    def set_scalar(self, value):
        """
        Sets the scalar value to a different value if needed.

        :param value: the new scalar value for the scaling object.
        :type value: float
        """
        if isinstance(value, numbers.Real):
            self._scalar = value
        else:
            raise self.ScalingTypeError(f"Received an invalid number \"{value}\" when setting the "+
                                        f"scalar value in Scaling object scaling \"{self.field}\"")
    @property
    def scalar(self):
        """
        :return: the scaling value for the scaling object.
        """
        return self._scalar

    @property
    def offset(self):
        """
        :return: the offset value for the scaling object.
        """
        return self._offset


class ScalingCollection(ContainerCollectionBase):
    """
    A collection of :class:`~matcal.core.data.Scaling` objects. This is 
    used to combine multiple scaling objects so that
    more than one scaling function or value can be applied to a data set. 
    This class is used when applying different
    scaling functions or values to different fields within a data set.
    """
    _collection_type = Scaling

    def __init__(self, name, *scalings):
        """
        :param name: the name for the scaling collection used for identification for error catching.
        :type name: str

        :param scalings: The scaling items to be added to the collection. They 
            can be passed in as comma separated
            list or an unpacked list. Unpack a list using \\*list_name.
        :type scalings: list(:class:`~matcal.core.data.Scaling`)

        :raises CollectionValueError: If name is an empty string.
        :raises CollectionTypeError: If name is not a string and the 
            scalings to be added to the collection are
            not of the correct type.
        """
        super().__init__(name, *scalings)

    def add(self, scaling):
        """
        Adds a :class:`~matcal.core.data.Scaling` object to the
        collection.

        :param scaling: scaling object to be added to the collection
        :type scaling: :class:`~matcal.core.data.Scaling`
        """
        self._check_item_is_correct_type(scaling)
        super().add(scaling.field, scaling)


class DataConditionerBase(ABC):
    """
    This is the base class for MatCal data conditioners. The data conditioners
    attempt to modify all data sets for a state in a single evaluation set such that
    the experimental data is on the order of -1 to 1. The data is modified 
    according to:

    .. math::
        \\mathbf{d}_c = \\frac{\\mathbf{d}-o}{s}

    where :math:`\\mathbf{d}` is a vector created from all data sets included in a single state, 
    :math:`o` is a scalar data offset calculated from :math:`\\mathbf{d}`, and :math:`s` is 
    a scalar scale factor calculated from :math:`d`. 
    If :math:`s=0` after it is calculated, the base conditioner class will 
    change the scale factor such 
    that :math:`s=mean\\left(\\left|\\mathbf{d}\\right|\\right)` or the 
    average of the absolute value of the relevant data. 
    If :math:`s` is still near zero, then the vector is full of zero or near zero values  and 
    the base conditioner sets the scale factor to :math:`s=1`

    The calculation of :math:`o` and :math:`s` is specific to the derived 
    conditioner class. The abstract methods 
    :meth:`~matcal.core.data.DataConditionerBase.get_scale_for_data_field`
    and :meth:`~matcal.core.data.DataConditionerBase.get_scale_for_data_field`
    define the calculations
    for :math:`o` and :math:`s`. A custom user class can be defined to implement 
    conditioning of the user's choice by including only the implementation of these 
    methods. 
    """
    def __init__(self):
        self._zero_tolerance = 1e-14
        self._field_names = []
        self._field_offsets = OrderedDict()
        self._field_scales = OrderedDict()
        self._initialized = False

    def apply_to_data(self, passed_data):
        """
        Apply the conditioner to a data set. This can be any data set and 
        does not need to be the one that was used to initialize the data set. 

        If a field name in a the data set passed to this method 
        was not in the data set used to 
        initialize the conditioner, the passed data field is returned unchanged.

        :param passed_data: a data set to be conditioned using an initialized conditioner.
        :type passed_data: :class:`~matcal.core.data.Data`
        """
        self._check_data(passed_data)
        self._verify_initialized()
        conditioned_data = self._condition_data(passed_data)
        return conditioned_data

    def _verify_initialized(self):
        if not self._initialized:
            raise RuntimeError("Cannot condition passed data. Conditioner is not initialized.")

    def _condition_data(self, passed_data):
        conditioned_data = self._initialize_conditioned_array(passed_data)
        for field_name in conditioned_data.field_names:
            logger.debug("Conditioning Field Name: {}".format(field_name))
            data_to_condition = conditioned_data[field_name] 
            if self._is_noise_field(field_name):
                field_name_no_noise = self._get_non_noise_field_name(field_name)
                conditioned_data[field_name] = self._apply_field_conditioning(field_name_no_noise,
                    data_to_condition, self._condition_noise)
            else:
                conditioned_data[field_name] = self._apply_field_conditioning(field_name, 
                    data_to_condition, self._condition_field)
        conditioned_data.set_state(passed_data.state)
        conditioned_data.set_name("conditioned "+ passed_data.name)
        return conditioned_data

    def _initialize_conditioned_array(self, passed_data):
        formats = []
        for field_name in passed_data.field_names:
            if passed_data.dtype[field_name] == np.dtype('int'):
                formats.append(float)
            else:
                formats.append(passed_data.dtype[field_name])
        updated_dtype = np.dtype({'names':passed_data.field_names, 'formats':formats})
        conditioned_data = passed_data.copy().astype(updated_dtype)    
        
        return conditioned_data
                
    def _apply_field_conditioning(self, field_name, passed_data, condition_func):
        if field_name in self._field_names:
            conditioned_field_data = condition_func(field_name, passed_data)
            return conditioned_field_data
        else:
            return passed_data
    
    def _get_non_noise_field_name(self, field_name):
        field_name_no_noise = "_".join(field_name.split("_")[:-1])
        return field_name_no_noise

    def _check_data(self, passed_data):
        if not isinstance(passed_data, Data):
            passed_type = type(passed_data)
            raise TypeError("Conditioner needs a class derived from the MatCal Data class. "
                            f"Passed object of type {passed_type}")

    def _is_noise_field(self, field_name):
        return field_name.split("_")[-1].lower() == "noise"

    def _condition_field(self, field_name, field_data):
        offset = self._field_offsets[field_name]
        scale = self._field_scales[field_name]
        conditioned_field_data = (field_data - offset) / scale
        return conditioned_field_data

    def _condition_noise(self, field_name, field_data):
        scale = self._field_scales[field_name]
        conditioned_field_data = (field_data) / scale
        return conditioned_field_data

    def initialize_data_conditioning_values(self, data_list):
        """
        Initialize the conditioner for a given list of data sets from 
        a single state of a data collection.

        :param: list of data sets to be used for conditioning. Generally passed 
            as a ``__getitem_`` of a state from a :class:`~matcal.core.data.DataCollection`.
        :param type: list(:class:`~matcal.core.data.Data`)
        """
        combined_data = combine_data_sets_in_data_list(data_list)
        for field_name, values in combined_data.items():
            if self._is_noise_field(field_name):
                continue
            self._field_names.append(field_name)
            self._save_data_conditioning_values_for_field(field_name, values)
        self._verify_valid_initialization()
        self._initialized = True

    def _verify_valid_initialization(self):
        if not self._field_names:
            raise ValueError("Initialization failed. Initialization data list likely empty.")
        
    def _save_data_conditioning_values_for_field(self, field_name, field_data):   
        self._field_offsets[field_name] = self.get_offset_for_data_field(field_data)
        scale = self.get_scale_for_data_field(field_data)
        if scale < self._zero_tolerance:
            scale = np.max(np.abs(field_data))
        if scale < self._zero_tolerance:
            scale = 1.0
        self._field_scales[field_name] = scale

    @abstractmethod
    def get_scale_for_data_field(self, field_data):
        """
        Calculates the scale factor :math:`s` for the data conditioner given 
        all values for a specific field name from the data collection 
        for a single state. This scale factor will be used to condition all 
        data with this state and field name when compared using an evaluation set. 

        :param field_data: all data for a specific field from a single state of 
            a data collection used to calculate an objective in an evaluation set.
        :type field_data: ArrayLike
        """

    @abstractmethod
    def get_offset_for_data_field(self, field_data):
        """
        Calculates the offset :math:`o` for the data conditioner given 
        all values for a specific field name from the data collection 
        for a single state. This offset will be used to condition all 
        data with this state and field name when compared using an evaluation set.

        :param field_data: all data for a specific field from a single state of 
            a data collection used to calculate an objective in an evaluation set.
        :type field_data: ArrayLike
        """

class ReturnPassedDataConditioner(DataConditionerBase):
    """
    This data conditioner will make no changes to the data sets 
    included in the evaluation set. Its scale and offset values are 
    given by :math:`s=1` and :math:`o=0`
    """
    def get_scale_for_data_field(self, field_data):
        return 1.0

    def get_offset_for_data_field(self, field_data):
        return 0.0
   

class RangeDataConditioner(DataConditionerBase):
    """
    This data conditioner will condition data such that each 
    field from the initializing data list is in the range of 
    0 to 1. To do so the scale and offset values are calculated 
    as :math:`s=max\\left(\\mathbf{d}\\right)-min\\left(\\mathbf{d}\\right)` and 
    :math:`o=min\\left(\\mathbf{d}\\right)`.
    """
    def _calculate_field_range(self, field_data):
        range = (np.max(field_data) - np.min(field_data))
        return range

    def get_scale_for_data_field(self, field_data):
        return self._calculate_field_range(field_data)

    def get_offset_for_data_field(self, field_data):
        return np.min(field_data)


class MaxAbsDataConditioner(DataConditionerBase):
    """
    This data conditioner will condition data such that each 
    field from the initializing data list is in the range of 
    -1 to 1. To do so, the scale values are calculated 
    as :math:`s=max\\left(\\left|\\mathbf{d}\\right|\\right)` and 
    :math:`o=0`. Note that this only guarantees the 
    data will be in the range of -1 to 1, it does not enforce 
    that the data spans the entirety of -1 to 1. 
    """
    def get_scale_for_data_field(self, field_data):
        return np.max(np.abs(field_data))

    def get_offset_for_data_field(self, field_data):
        return 0.0


class AverageAbsDataConditioner(DataConditionerBase):
    """
    This data conditioner will condition data such that each 
    field from the initializing data list is on the order of 
    -1 to 1. To do so, the scale values are calculated 
    as :math:`s=mean\\left(\\left|\\mathbf{d}\\right|\\right)` and 
    :math:`o=0`.
    Note that this likely puts the all data in the field 
    on the order of -1 to 1, but the data could be well outside 
    of this range depending on the values in the data. 
    """
    def get_scale_for_data_field(self, field_data):
        return np.average(np.abs(field_data))

    def get_offset_for_data_field(self, field_data):
        return 0.0


def combine_data_sets_in_data_list(data_list):
    """
    Given a list of :class:`~matcal.core.data.Data` objects, 
    this function will return a dictionary where each 
    item is all values from the same field in from all data sets and 
    the key for the items are the field names.

    :param data_list: list of data sets that will be combined.
    :type data_list: list(:class:`~matcal.core.data.Data`)
    """
    combined_data = OrderedDict()
    for data in data_list:
        for field_name in data.field_names:
            field_data = data[field_name]
            if not field_name in combined_data.keys():
                combined_data[field_name] = field_data
            else:
                combined_data[field_name] = np.append(combined_data[field_name], field_data)
    return combined_data


def _scale_data(scaling_collection, data):
    scaled_data = data.copy()
    for field_name in data.field_names:
        if field_name in scaling_collection.keys():
            for scale in scaling_collection[field_name]:
                scaled_data = scale.apply_to_data(scaled_data)
    return scaled_data


def scale_data_collection(data_collection, field_name, scale, offset=0):
    """
    Scales all data sets in a data collection that have 
    the requested field. It will apply the correct 
    scale factor and offset to each data set and return 
    a new data collection that is scaled. Note that if 
    both are used, the offset is applied first and then 
    the results are scaled. A new scaled data collection 
    is returned and the old one is unmodified.

    :param data_collection: the data collection to be scaled
    :type data_collection: :class:`~matcal.core.data.DataCollection`

    :param field_name: the name of the field to be modified
    :type fied_name: str

    :param scale: a linear scale factor to scale the field
    :type scale: float

    :param offset: a constant offset to be added to the field
    :type offset: float

    :return: new scaled data collection
    :rtype: :class:`~matcal.core.data.DataCollection`
    """
    _check_type(data_collection, DataCollection, "data collection to be scaled")
    _check_type(field_name, str, "the field name to be scaled in the data collection")
    _check_type(scale, numbers.Real, "the scale factor to be applied to the data collection field")
    _check_type(offset, numbers.Real, "the offset to be applied to the data collection field")

    name = "scale_{}".format(field_name)
    scaling_collection = ScalingCollection(name, Scaling(field_name, scale, offset))
    scaled_data_collection = DataCollection(name+"_{}".format(data_collection.name))
    for state in data_collection.keys():
        for data in data_collection[state]:
            scaled_data_collection.add(_scale_data(scaling_collection, data))
    return scaled_data_collection


def _check_type(variable, desired_type, message):
    if not isinstance(variable, desired_type):
        raise Data.TypeError(f"The {message} is not the correct type. Expected type {desired_type} "
                             f" and recieved {type(variable)}. Check input.")


def convert_data_to_dictionary(data):
    """
    Converts a MatCal :class:`~matcal.core.data.Data`
    class into a dictionary of np.arrays.

    :param data: a MatCal data set
    :type data: :class:`~matcal.core.data.Data`

    :return: dictionary conversion of the data object
    :rtype: OrderedDict
    """

    if not isinstance(data, Data):
        raise TypeError(f"The object passed to be converted to a dictionary" 
                         f" must be a MatCal Data type. Received an object of type {type(data)}.")

    d = OrderedDict()
    for key in list(data.field_names):
        kdata = data[key]
        if isinstance(kdata, Data):
            kdata = np.asarray(kdata)
        d[key] = kdata
    return d


def convert_dictionary_to_data(dict_data):
    """
    Takes a dictionary and attempts to create a
    MatCal :class:`~matcal.core.data.Data` object.
    The keys for the dictionary are expected to be 
    strings for the field names and the values 
    are expected to be valid numeric or string data. 

    :param dict_data: a dictionary with field names as keys and 
       the data values as the dictionary values.
    :type dict_data: dict or OrderedDict

    :return: a Data object with the default state :class:`~matcal.core.state.SolitaryState`. 
    :rtype: :class:`~matcal.core.data.Data`
    """
    _check_dictionary_data(dict_data)
    data = _create_array_from_dict(dict_data)
    return Data(data)


def _check_dictionary_data(dict_data):
    for value in dict_data.values():
        if value is None:
            raise Data.TypeError("Attempting to put a None in a Data object")


def _create_array_from_dict(dict_data):
    row_data = []
    data_types = []
    first_dim_length = None
    for key, item in dict_data.items():
        item = np.atleast_1d(np.array(item))
        first_dim_length = _set_first_dim(first_dim_length, item.shape)
        if not _confirm_first_dimension_length(first_dim_length, item.shape):
            raise UnequalTimeDimensionSizeError(key)
        data_types.append(_determine_data_type(item, key))
        row_data.append(item)
    converted_array = np.core.records.fromarrays(row_data, dtype=data_types)
    return converted_array


def _determine_data_type(item, key):
    dtype=item.dtype
   
    if np.issubdtype(item.dtype, np.integer):
        dtype=float
    else:
        dtype=dtype
    if item.ndim <= 1:
        dtype = (key, dtype)
        return dtype
    else:
        return (key, dtype, item.shape[1:])


class UnequalTimeDimensionSizeError(RuntimeError):

    def __init__(self, key_name):
        message = f"Field: {key_name}\n Does not have the same first (time) dimension length"
        super().__init__(message)


def _confirm_first_dimension_length(ref_size, data_shape):
    return ref_size == data_shape[0]


def _set_first_dim(old_first_dim, data_shape):
    if old_first_dim is None:
        return data_shape[0]
    else:
        return old_first_dim


def _serialize_data(data_to_serialize:Data)->dict:
    out_dict = convert_data_to_dictionary(data_to_serialize)
    for key, value in out_dict.items():
        out_dict[key] = _format_serial(value)
    return out_dict