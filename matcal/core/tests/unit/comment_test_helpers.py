from matcal.core.constants import TIME_KEY


def assert_comment_line_present(test_case, text, expected_line):
    test_case.assertIn(f"# {expected_line}", text)


def assert_source_fields_comment(test_case, text, dependent_field, uses_time=True):
    if uses_time:
        expected = (
            f'Using tabulated "{TIME_KEY}" and "{dependent_field}" '
            f'fields from source data set.'
        )
    else:
        expected = (
            f'Using "{dependent_field}" field from source data set; '
            f'no "{TIME_KEY}" field was provided.'
        )
    assert_comment_line_present(test_case, text, expected)


def assert_source_collection_comment(test_case, text, collection_name="boundary conditions"):
    assert_comment_line_present(
        test_case,
        text,
        f'Source data collection: "{collection_name}".',
    )


def assert_data_set_index_comment(test_case, text, index=0):
    assert_comment_line_present(
        test_case,
        text,
        f"Selected data set index: {index}.",
    )


def assert_data_set_name_comment(test_case, text, data_set_name):
    assert_comment_line_present(
        test_case,
        text,
        f'Selected data set name: "{data_set_name}".',
    )


def assert_selection_reason_comment(test_case, text, field_key, state_name):
    test_case.assertIn(
        f'# Selected this data set because its absolute maximum "{field_key}" value of ',
        text,
    )
    test_case.assertIn(
        f'is the largest among data sets for state "{state_name}".',
        text,
    )


def assert_rate_ramp_comment(test_case, text, field_key, rate_key):
    test_case.assertIn(
        f'# Constructed a 2-point linear ramp from the maximum absolute "{field_key}" '
        f'value using "{rate_key}" = ',
        text,
    )


def assert_default_ramp_comment(test_case, text, field_key):
    test_case.assertIn('# Constructed default points (0, 0) and (1, ', text)
    test_case.assertIn(
        f'from the maximum absolute "{field_key}" value because no "{TIME_KEY}" '
        f'field or compatible rate parameter was provided.',
        text,
    )


def assert_symmetry_displacement_comment(test_case, text):
    assert_comment_line_present(
        test_case,
        text,
        "Applied symmetry factor of 0.5 to the prescribed displacement.",
    )


def assert_symmetry_rotation_comment(test_case, text):
    assert_comment_line_present(
        test_case,
        text,
        "Applied symmetry factor of 0.5 to the prescribed rotation.",
    )


def assert_rotation_to_radians_comment(test_case, text):
    assert_comment_line_present(
        test_case,
        text,
        "Converted rotation from degrees to radians.",
    )


def assert_rotation_rename_comment(test_case, text, rotation_field, displacement_field):
    assert_comment_line_present(
        test_case,
        text,
        f'Renamed "{rotation_field}" to "{displacement_field}" for the SIERRA prescribed displacement boundary condition.',
    )


def assert_metadata_common_fields(
    test_case,
    metadata,
    field_key,
    state_name,
    data_collection_name,
    selected_data_set_index=None,
    selected_data_set_name=None,
):
    test_case.assertEqual(metadata["field_key"], field_key)
    test_case.assertEqual(metadata["state_name"], state_name)
    test_case.assertEqual(metadata["data_collection_name"], data_collection_name)

    if selected_data_set_index is not None:
        test_case.assertEqual(metadata["selected_data_set_index"], selected_data_set_index)

    if selected_data_set_name is not None:
        test_case.assertEqual(metadata["selected_data_set_name"], selected_data_set_name)