"""
This module contains all data related to MatCal 
state parameters and all classes are meant for users.
"""

from collections import OrderedDict
from numbers import Number

from matcal.core.utilities import CollectionBase, matcal_name_format

class State(object):
    """
    The MatCal object that stores state information tied to a data set. State information is passed to the model at run
    time. State information can include model parameters or boundary condition information, but is intended to 
    be a parameter/quantity that is unique to the experiment setup and needed to successfully simulate the experiment. 
    """

    class KeyError(RuntimeError):
        pass

    def __init__(self, name, **kwargs):
        """
    :param name: the name for this state. This must be a string that is valid for a directory name.
    :type name: str

    :param kwargs: comma-delimited list of name/value pairs for each state variable (e.g., rate=1e-4,
        temperature=298, direction='x')
    :type kwargs: dict(str, float or str)
    """
        self._name = None
        self.set_name(name)
        self._state_variables = OrderedDict(**kwargs)
        self._solitary_state = False
        self._check_state_variables()

    def update(self, update_dict):
        """
        Updates the state using a dictionary where 
        state variable names are the keys, and the corresponding
        values are the assigned state variable values. This can 
        be used to update existing state variables or add new.

        :param update_dict: the dictionary of values to be added to the state.
        :type update_dict: dict
        """
        self._state_variables.update(update_dict)
        self._check_state_variables()

    def update_state_variable(self, state_var_name, value):
        """
        Updates or adds a new specific state variable in a state.

        :param state_var_name: state variable name
        :type state_var_name: str

        :param value: state variable name
        :type value: str or float

        """
        self.update({state_var_name:value})

    def set_name(self, name):
        self._name = self._check_and_format_name(name)

    @property
    def name(self):
        """
        :return: the state name
        :rtype: str
        """        
        return self._name

    @property
    def params(self):
        """
        :return: the state parameters and values
        :rtype: dict
        """
        return self._state_variables

    @property
    def solitary_state(self):
        """
        :return: true if it has no parameters, false if parameters are specified
        :rtype: bool
        """
        return self._solitary_state

    def __getitem__(self, item):
        if self._solitary_state:
            return None
        if item not in self._state_variables.keys():
            raise self.KeyError("The state variable \"{}\" is not defined for state \"{}\"".format(item, self._name))

        return self._state_variables[item]

    def __eq__(self, other):
        check = isinstance(other, State) and self.name == other.name
        check = check and self.solitary_state == other.solitary_state
        return check and self.params == other.params

    def __hash__(self):
        return hash(self.name)

    def _check_state_variables(self):
        for key, val in self._state_variables.items():
            if not isinstance(key, str):
                raise TypeError(f"The state variable name  \"{key}\" is not a valid state variable name. "
                    f"It must be a string. It is of type {type(key)}")
            if not isinstance(val, (str, Number)):
                raise TypeError(f"The state variable value  \"{val}\" is not a valid state variable value. "
                    f"It must be a string or a numeric value. It is of type {type(val)}")

    def _check_and_format_name(self, name):
        if not isinstance(name, str):
            raise TypeError(f"The state name \"{name}\" is not a valid state variable name. "
                f"It must be a string. It is of type {type(name)}")
        return matcal_name_format(name)

                            

class SolitaryState(State):
    """
    The default state for MatCal if no state is specified by the user.
    """

    def __init__(self):
        super().__init__("matcal_default_state")
        self._solitary_state = True

class StateCollection(CollectionBase):
    """
    A collection of :class:`~matcal.core.state.State` objects. This is used to combine multiple state
    objects so that they can be passed to a MatCal study. MatCal will use all states in the state collections
    as the study states.
    """
    _collection_type = State

    def __init__(self, name, *states):
        """
        :param name: The name of the state collection.
        :type name: str

        :param states: the states to be added to the collection.
        :type states: list(:class:`~matcal.core.state.State` objects).

        :raises CollectionValueError: If name is a an empty string.
        :raises CollectionTypeError: If name is not a string and the states to be added to the collection are
            not of the correct type.
         """
        super().__init__(name, *states)
