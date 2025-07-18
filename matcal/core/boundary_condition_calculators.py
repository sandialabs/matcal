import numpy as np
from copy import deepcopy

from matcal.core.constants import ENG_STRAIN_KEY, STRAIN_RATE_KEY, TEMPERATURE_KEY,\
TRUE_STRAIN_KEY, DISPLACEMENT_KEY, DISPLACEMENT_RATE_KEY, ROTATION_KEY, ROTATION_RATE_KEY,\
TIME_KEY

from matcal.core.data import convert_data_to_dictionary, convert_dictionary_to_data


class BoundaryConditionDeterminationError(RuntimeError):
    """"""
def max_state_values(data_list, data_key):
    max_value = None
    max_data_index = None
    max_value_index = None
    for index, current_data in enumerate(data_list):
        current_value = np.max(np.abs(current_data[data_key]))
        if max_value is None:
            max_value = current_value
            max_data_index = index
            max_value_index = np.argmax(np.abs(current_data[data_key]))
        elif current_value > max_value:
            max_value = current_value
            max_data_index =index
            max_value_index = np.argmax(np.abs(current_data[data_key]))

    return max_data_index, data_list[max_data_index], max_value_index

def _get_max_abs_value_index_for_field(data, key):
    return np.argmax(np.abs(data[key]))


def get_field_function_from_data(data, state,  
                                 compatible_rate_key, field_key, 
                                 params_by_precedent={}, scale_factor=None):
    if scale_factor is None:
        scale_factor = 1
    function = None 
    params=params_by_precedent
    if state.params or params_by_precedent:
        params = dict(state.params)
        params.update(params_by_precedent)
    max_value_index = _get_max_abs_value_index_for_field(data, field_key)
    max_value = data[field_key][max_value_index]
    if TIME_KEY in data.field_names:
        function = np.array([data[TIME_KEY], data[field_key]*scale_factor]).T
    elif compatible_rate_key in params:
        rate = params[compatible_rate_key]
        function =  np.array([[0, 0], [max_value/rate*np.sign(max_value), max_value*scale_factor]])
    else:
        function =  np.array([[0, 0], [1, max_value*scale_factor]])
   
    function_dict = {TIME_KEY:function[:,0],field_key:function[:,1]}
    
    return convert_dictionary_to_data(function_dict)


def convert_true_strain_to_eng_strain(true_strain):
    return np.exp(true_strain) - 1


def add_engineering_strain_to_data_collection_state_from_true_strain(data_collection, state):
    for index, data in enumerate(data_collection[state.name]):
        if TRUE_STRAIN_KEY in data.field_names and ENG_STRAIN_KEY not in data.field_names:
            data_dict = convert_data_to_dictionary(data)
            data_dict[ENG_STRAIN_KEY] = convert_true_strain_to_eng_strain(data[TRUE_STRAIN_KEY]) 
            data_collection[state.name][index] = convert_dictionary_to_data(data_dict)
            data_collection[state.name][index].set_state(state)
    return data_collection


def get_displacement_function_from_strain_data_collection(data_collection, state, 
                                                          params_by_precedent={}, 
                                                          scale_factor=None, 
                                                          convert_true_strain=True):
    data_collection_for_function = deepcopy(data_collection)
    if convert_true_strain:
        add_eng_strain_to_data = add_engineering_strain_to_data_collection_state_from_true_strain
        data_collection_for_function = add_eng_strain_to_data(data_collection_for_function, state)
    func = get_field_function_from_data_collection(data_collection_for_function, state, 
                                                   params_by_precedent, 
                                                   scale_factor, ENG_STRAIN_KEY, STRAIN_RATE_KEY, 
                                                   [ENG_STRAIN_KEY])
    func.rename_field(ENG_STRAIN_KEY, DISPLACEMENT_KEY)
    return func


def get_displacement_function_from_load_displacement_data_collection(data_collection, state, 
                                                                     params_by_precedent={}, 
                                                                     scale_factor=None):
    return get_field_function_from_data_collection(data_collection, state, params_by_precedent, 
                                                   scale_factor, 
                                                   DISPLACEMENT_KEY, DISPLACEMENT_RATE_KEY, 
                                                   [DISPLACEMENT_KEY])


def get_temperature_function_from_data_collection(data_collection, state, params_by_precedent={},  
                                                  temperature_key=TEMPERATURE_KEY, 
                                                  scale_factor=None):
    return get_field_function_from_data_collection(data_collection, state, params_by_precedent, 
                                                   scale_factor, 
                                                   temperature_key, None, 
                                                   [temperature_key, TIME_KEY])


def get_rotation_function_from_data_collection(data_collection, state, params_by_precedent={}, 
                                               scale_factor=None):
    return get_field_function_from_data_collection(data_collection, state, params_by_precedent, 
                                                   scale_factor, 
                                                   ROTATION_KEY, ROTATION_RATE_KEY, [ROTATION_KEY])


def get_field_function_from_data_collection(data_collection, state, params_by_precedent, 
                                            scale_factor, 
                                            field_key, rate_key, required_bc_keys: list):
    _verify_state_in_data_collection(state, data_collection)
    res = _verify_required_keys_in_all_data_collection(data_collection[state.name],
                                                       required_bc_keys)
    required_fields_in_state_data_sets, required_fields_string = res
    if not required_fields_in_state_data_sets:
        raise_required_fields_not_found_error(state, required_fields_string, data_collection.name)
    max_data_index, max_data_set, max_value_index = max_state_values(data_collection[state.name], 
                                                                     field_key)
    function = get_field_function_from_data(max_data_set, state, 
                                            rate_key, field_key, params_by_precedent, 
                                            scale_factor)
    return function


def _verify_state_in_data_collection(state, data_collection):
    if state.name not in data_collection.state_names:
        raise BoundaryConditionDeterminationError(f"The data collection \"{data_collection.name}\" " +  
                                                  f"does not have state \"{state.name}\"")


def _verify_required_keys_in_all_data_collection(data_list, required_bc_keys):
    required_fields = ""
    for required_key in required_bc_keys:
        has_valid_key_in_all_state_data = False
        for data in data_list:
            if required_key in data.field_names:
                has_valid_key_in_all_state_data = True
            else:
                has_valid_key_in_all_state_data = False
        required_fields += f"\"{required_key}\"\n"
    return has_valid_key_in_all_state_data, required_fields


def raise_required_fields_not_found_error(state, required_fields_str, data_collection_name):
    raise BoundaryConditionDeterminationError(f"The data sets for state \"{state.name}\" do not \n"
        "all have the required fields for boundary condition specification.\n"
        "Check the data passed to the \"add_boundary_condition_data\" method "
        "for the model and verify it was input correctly.\n"
        "Required fields for each data set in this state are:"
        f"\n{required_fields_str}\n" 
        "All datasets for each state in the data collection must have the "
        "same required field for boundary condition determination. Mixing field names "
        " for boundary condition determination within a state can cause this error. "
        f"Check DataCollection with name \"{data_collection_name}\""
        f" for state \"{state.name}\". ")
