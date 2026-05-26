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
            max_data_index = index
            max_value_index = np.argmax(np.abs(current_data[data_key]))

    return max_data_index, data_list[max_data_index], max_value_index


def _get_max_abs_value_index_for_field(data, key):
    return np.argmax(np.abs(data[key]))


def _get_data_name(data, default=None):
    if hasattr(data, "name"):
        data_name = data.name
        if data_name is not None and data_name != "":
            return data_name
    return default


def _get_source_method_comment(field_key, has_time):
    if has_time:
        return f'Using tabulated "{TIME_KEY}" and "{field_key}" fields from source data set.'
    return f'Using "{field_key}" field from source data set; no "{TIME_KEY}" field was provided.'


def _build_field_function_metadata(
    data,
    state,
    compatible_rate_key,
    field_key,
    params_by_precedent=None,
    scale_factor=None,
):
    if params_by_precedent is None:
        params_by_precedent = {}
    if scale_factor is None:
        scale_factor = 1

    params = params_by_precedent
    if state.params or params_by_precedent:
        params = dict(state.params)
        params.update(params_by_precedent)

    max_value_index = _get_max_abs_value_index_for_field(data, field_key)
    max_value = data[field_key][max_value_index]
    used_time_history = TIME_KEY in data.field_names

    metadata = {
        "state_name": state.name,
        "field_key": field_key,
        "compatible_rate_key": compatible_rate_key,
        "scale_factor": scale_factor,
        "max_value_index": int(max_value_index),
        "max_value": max_value,
        "used_time_history": used_time_history,
        "used_rate": False,
        "rate_value": None,
        "constructed_end_time": None,
        "source_method_comment": _get_source_method_comment(field_key, used_time_history),
    }

    if (
        not used_time_history and
        compatible_rate_key is not None and
        compatible_rate_key in params
    ):
        rate = params[compatible_rate_key]
        metadata["used_rate"] = True
        metadata["rate_value"] = rate
        metadata["constructed_end_time"] = max_value / rate * np.sign(max_value)

    return metadata


def format_bc_function_comment_lines(metadata):
    field_key = metadata["field_key"]
    lines = []

    lines.append(metadata["source_method_comment"])

    if metadata["used_rate"]:
        lines.append(
            f'Constructed a 2-point linear ramp from the maximum absolute "{field_key}" value '
            f'using "{metadata["compatible_rate_key"]}" = {metadata["rate_value"]}.'
        )
        lines.append(
            f'Constructed points: (0, 0) and '
            f'({metadata["constructed_end_time"]}, {metadata["max_value"] * metadata["scale_factor"]}).'
        )
    elif not metadata["used_time_history"]:
        lines.append(
            f'Constructed default points (0, 0) and '
            f'(1, {metadata["max_value"] * metadata["scale_factor"]}) from the maximum '
            f'absolute "{field_key}" value because no "{TIME_KEY}" field or compatible rate '
            f'parameter was provided.'
        )

    if metadata.get("data_collection_name") is not None:
        lines.append(f'Source data collection: "{metadata["data_collection_name"]}".')

    if metadata.get("selected_data_set_index") is not None:
        lines.append(f'Selected data set index: {metadata["selected_data_set_index"]}.')

    if metadata.get("selected_data_set_name") is not None:
        lines.append(f'Selected data set name: "{metadata["selected_data_set_name"]}".')

    lines.append(
        f'Selected this data set because its absolute maximum "{field_key}" value of '
        f'{metadata["max_value"]} at index {metadata["max_value_index"]} is the largest '
        f'among data sets for state "{metadata["state_name"]}".'
    )

    return lines


def get_field_function_from_data(
    data,
    state,
    compatible_rate_key,
    field_key,
    params_by_precedent={},
    scale_factor=None,
    return_metadata=False,
):
    if scale_factor is None:
        scale_factor = 1
    function = None
    params = params_by_precedent
    if state.params or params_by_precedent:
        params = dict(state.params)
        params.update(params_by_precedent)

    max_value_index = _get_max_abs_value_index_for_field(data, field_key)
    max_value = data[field_key][max_value_index]

    if TIME_KEY in data.field_names:
        function = np.array([data[TIME_KEY], data[field_key] * scale_factor]).T
    elif compatible_rate_key in params:
        rate = params[compatible_rate_key]
        function = np.array([
            [0, 0],
            [max_value / rate * np.sign(max_value), max_value * scale_factor]
        ])
    else:
        function = np.array([[0, 0], [1, max_value * scale_factor]])

    function_dict = {TIME_KEY: function[:, 0], field_key: function[:, 1]}
    function_data = convert_dictionary_to_data(function_dict)

    if return_metadata:
        metadata = _build_field_function_metadata(
            data,
            state,
            compatible_rate_key,
            field_key,
            params_by_precedent=params_by_precedent,
            scale_factor=scale_factor,
        )
        return function_data, metadata

    return function_data


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


def get_displacement_function_from_strain_data_collection(
    data_collection,
    state,
    params_by_precedent={},
    scale_factor=None,
    convert_true_strain=True,
    return_metadata=False,
):
    data_collection_for_function = deepcopy(data_collection)
    converted_true_strain = False
    if convert_true_strain:
        add_eng_strain_to_data = add_engineering_strain_to_data_collection_state_from_true_strain
        data_collection_for_function = add_eng_strain_to_data(data_collection_for_function, state)
        converted_true_strain = True

    result = get_field_function_from_data_collection(
        data_collection_for_function,
        state,
        params_by_precedent,
        scale_factor,
        ENG_STRAIN_KEY,
        STRAIN_RATE_KEY,
        [ENG_STRAIN_KEY],
        return_metadata=return_metadata,
    )

    if return_metadata:
        func, metadata = result
        func.rename_field(ENG_STRAIN_KEY, DISPLACEMENT_KEY)
        metadata["input_kind"] = "strain"
        metadata["source_field"] = ENG_STRAIN_KEY
        metadata["renamed_to"] = DISPLACEMENT_KEY
        metadata["converted_true_strain"] = converted_true_strain
        return func, metadata

    func = result
    func.rename_field(ENG_STRAIN_KEY, DISPLACEMENT_KEY)
    return func


def get_displacement_function_from_load_displacement_data_collection(
    data_collection,
    state,
    params_by_precedent={},
    scale_factor=None,
    return_metadata=False,
):
    result = get_field_function_from_data_collection(
        data_collection,
        state,
        params_by_precedent,
        scale_factor,
        DISPLACEMENT_KEY,
        DISPLACEMENT_RATE_KEY,
        [DISPLACEMENT_KEY],
        return_metadata=return_metadata,
    )
    if return_metadata:
        func, metadata = result
        metadata["input_kind"] = "displacement"
        metadata["source_field"] = DISPLACEMENT_KEY
        return func, metadata
    return result


def get_temperature_function_from_data_collection(
    data_collection,
    state,
    params_by_precedent={},
    temperature_key=TEMPERATURE_KEY,
    scale_factor=None,
    return_metadata=False,
):
    result = get_field_function_from_data_collection(
        data_collection,
        state,
        params_by_precedent,
        scale_factor,
        temperature_key,
        None,
        [temperature_key, TIME_KEY],
        return_metadata=return_metadata,
    )
    if return_metadata:
        func, metadata = result
        metadata["input_kind"] = "temperature"
        metadata["source_field"] = temperature_key
        return func, metadata
    return result


def get_rotation_function_from_data_collection(
    data_collection,
    state,
    params_by_precedent={},
    scale_factor=None,
    return_metadata=False,
):
    result = get_field_function_from_data_collection(
        data_collection,
        state,
        params_by_precedent,
        scale_factor,
        ROTATION_KEY,
        ROTATION_RATE_KEY,
        [ROTATION_KEY],
        return_metadata=return_metadata,
    )
    if return_metadata:
        func, metadata = result
        metadata["input_kind"] = "rotation"
        metadata["source_field"] = ROTATION_KEY
        return func, metadata
    return result


def get_field_function_from_data_collection(
    data_collection,
    state,
    params_by_precedent,
    scale_factor,
    field_key,
    rate_key,
    required_bc_keys: list,
    return_metadata=False,
):
    _verify_state_in_data_collection(state, data_collection)
    res = _verify_required_keys_in_all_data_collection(
        data_collection[state.name],
        required_bc_keys,
    )
    required_fields_in_state_data_sets, required_fields_string = res
    if not required_fields_in_state_data_sets:
        raise_required_fields_not_found_error(state, required_fields_string, data_collection.name)

    max_data_index, max_data_set, max_value_index = max_state_values(
        data_collection[state.name],
        field_key,
    )

    if return_metadata:
        function, metadata = get_field_function_from_data(
            max_data_set,
            state,
            rate_key,
            field_key,
            params_by_precedent,
            scale_factor,
            return_metadata=True,
        )
        metadata["data_collection_name"] = data_collection.name
        metadata["selected_data_set_index"] = int(max_data_index)
        metadata["selected_data_set_name"] = _get_data_name(max_data_set)
        return function, metadata

    function = get_field_function_from_data(
        max_data_set,
        state,
        rate_key,
        field_key,
        params_by_precedent,
        scale_factor,
    )
    return function


def _verify_state_in_data_collection(state, data_collection):
    if state.name not in data_collection.state_names:
        raise BoundaryConditionDeterminationError(f"The data collection \"{data_collection.name}\" " +
                                                  f"does not have state \"{state.name}\"")


def _verify_required_keys_in_all_data_collection(data_list, required_bc_keys):
    required_fields = ""
    all_required_keys_present = True
    for required_key in required_bc_keys:
        has_valid_key_in_all_state_data = True
        for data in data_list:
            if required_key not in data.field_names:
                has_valid_key_in_all_state_data = False
                break
        all_required_keys_present = all_required_keys_present and has_valid_key_in_all_state_data
        required_fields += f"\"{required_key}\"\n"
    return all_required_keys_present, required_fields


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