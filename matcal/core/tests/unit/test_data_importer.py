import os
import glob
import numpy as np
from copy import copy
import csv
from unittest.mock import patch

from matcal.core.data_importer import (
    CSVDataImporter,
    DOSFileError,
    FileData,
    InvalidCharacterError,
    _has_dos_newlines,
    _is_dos,
    _has_invalid_lines,
    _report_invalid_utc_lines,
)
from matcal.core.data_importer import NumpyDataImporter, BatchDataImporter
from matcal.core.data import convert_dictionary_to_data, Data
from matcal.core.state import State, SolitaryState
from matcal.core.tests.MatcalUnitTest import MatcalUnitTest

TEST_REFERENCE_DIR = os.path.join(
    os.path.dirname(__file__), "test_reference", "data_importer"
)


def read_csv(filename, header=None):
    with open(filename, "r") as csvfile:

        if header is None:
            c = csv.DictReader(csvfile)
            keys = c.fieldnames
        else:
            keys = header
            c = csv.reader(csvfile)
        csv_dict = {}
        for key in keys:
            csv_dict[key] = []

        for row in c:
            for idx, key in enumerate(keys):
                if header is None:
                    lookup = key
                else:
                    lookup = idx
                csv_dict[key].append(float(row[lookup]))
        for key in keys:
            csv_dict[key] = np.array(csv_dict[key])
    return csv_dict


class CSVDataImporterTest(MatcalUnitTest):

    def setUp(self):
        super().setUp(__file__)

    @classmethod
    def setUpClass(cls) -> None:
        cls.default_csv_file = os.path.join(TEST_REFERENCE_DIR, "default.csv")
        cls.headerless_csv_file = os.path.join(TEST_REFERENCE_DIR, "headerless.csv")
        cls.state_header_csv_file = os.path.join(TEST_REFERENCE_DIR, "state_header.csv")
        cls.dashed_header_file = os.path.join(
            TEST_REFERENCE_DIR, "data_with_dashes.csv"
        )

    # --- CSVDataImporter-only tests remain here ---

    def test_non_existent_file_will_throw_file_not_found_error(self):
        with self.assertRaises(FileNotFoundError):
            CSVDataImporter("invalid_file.csv")

    def test_directory_path_raises_FileNotFoundError_with_directory_message(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(FileNotFoundError) as ctx:
                CSVDataImporter(tmpdir)
            self.assertIn("directory", str(ctx.exception).lower())

    def test_invalid_filename_type(self):
        with self.assertRaises(TypeError):
            CSVDataImporter(1)

    def test_import_header_with_dashes(self):
        d_imp = CSVDataImporter(self.dashed_header_file)
        data = d_imp.load()
        goal_fields = ["time", "field-A", "field-1", "field-a"]
        test_fields = data.field_names
        self.assertEqual(len(goal_fields), len(test_fields))
        for goal in goal_fields:
            self.assertIn(goal, test_fields)

    def test_non_interpretable_file_will_throw_data_format_error(self):
        d_nan = CSVDataImporter(
            os.path.join(TEST_REFERENCE_DIR, "nan_test.csv"), import_strings=True
        )
        with self.assertRaises(ValueError):
            d_nan.load()

        d_inf = CSVDataImporter(os.path.join(TEST_REFERENCE_DIR, "inf_test.csv"))
        with self.assertRaises(ValueError):
            d_inf.load()

        d_str = CSVDataImporter(os.path.join(TEST_REFERENCE_DIR, "str_test.csv"))
        with self.assertRaises(TypeError):
            d_str.load()

        d_char = CSVDataImporter(os.path.join(TEST_REFERENCE_DIR, "char_test.csv"))
        with self.assertRaises(TypeError):
            d_char.load()

        d_test = CSVDataImporter(
            os.path.join(TEST_REFERENCE_DIR, "non_interpretable_data.csv")
        )
        with self.assertRaises(TypeError):
            d_test.load()

    def test_non_finite_data_raises_value_error(self):
        d_inf = CSVDataImporter(os.path.join(TEST_REFERENCE_DIR, "inf_test.csv"))
        with self.assertRaises(ValueError):
            d_inf.load()

    def test_drop_NaNs(self):
        d_str = CSVDataImporter(
            os.path.join(TEST_REFERENCE_DIR, "data_with_nans_infs.csv")
        )
        with self.assertRaises(ValueError):
            d_str.load()

        d_str = CSVDataImporter(
            os.path.join(TEST_REFERENCE_DIR, "data_with_nans_infs.csv"), drop_NaNs=True
        )
        data = d_str.load()

        data_gold = np.genfromtxt(
            os.path.join(TEST_REFERENCE_DIR, "data_with_nans_infs.csv"),
            skip_header=1,
            delimiter=",",
        )
        data_gold = data_gold[np.isfinite(data_gold).all(axis=1), :]

        self.assert_close_arrays(data_gold[:, 0], data["load"])
        self.assert_close_arrays(data_gold[:, 1], data["displacement"])

    def test_drop_NaNs_skips_string_columns(self):
        string_col_file = os.path.join(TEST_REFERENCE_DIR, "string_col_with_nans.csv")
        importer = CSVDataImporter(string_col_file, import_strings=True, drop_NaNs=True)
        data = importer.load()
        self.assert_close_arrays(np.array([1.0, 4.0]), data["load"])
        self.assert_close_arrays(np.array([2.0, 6.0]), data["displacement"])

    def test_read_strings(self):
        d_str = CSVDataImporter(
            os.path.join(TEST_REFERENCE_DIR, "str_test.csv"), import_strings=True
        )
        data = d_str.load()
        self.assertTrue("e" in data["load"])

        d_char = CSVDataImporter(
            os.path.join(TEST_REFERENCE_DIR, "char_test.csv"), import_strings=True
        )
        data = d_char.load()
        self.assertTrue("&" in data["load"])

    def test_valid_default_file(self):
        d_0 = CSVDataImporter(self.default_csv_file)
        data = d_0.load()
        self.assertDefaultDataCorrect(data)
        self.assertEqual(data.name, os.path.abspath(self.default_csv_file))

    def test_empty_file(self):
        with open("empty.csv", "w") as f:
            f.write("X, Y\n")
        d0 = CSVDataImporter("empty.csv")
        with self.assertRaises(ValueError):
            d0.load()

    def test_empty_column(self):
        with open("empty.csv", "w") as f:
            f.write("X, Y, Z\n")
            f.write("0, , 0\n")
            f.write("0, , 0\n")

        d0 = CSVDataImporter("empty.csv")
        with self.assertRaises(TypeError):
            d0.load()

    def test_field_names(self):
        d_0 = CSVDataImporter(self.default_csv_file)
        data = d_0.load()
        self.assertListEqual(list(data.field_names), ["U", "F"])

    def test_get_length_of_data(self):
        d = CSVDataImporter(self.default_csv_file)
        data = d.load()
        self.assertEqual(data.length, 4)

    def test_load_data(self):
        d = CSVDataImporter(self.default_csv_file)
        data = d.load()
        self.assertIn("U", data.field_names)
        self.assertIn("F", data.field_names)
        self.assertEqual(data.length, 4)
        self.assertIsInstance(data, Data)
        self.assertEqual(data.name, self.default_csv_file)

    def test_state_header_values(self):
        d = CSVDataImporter(self.state_header_csv_file)
        data = d.load()
        refstate = {"rate": 4.0, "temperature": 300.0, "extra": 1.0, "str": "my_str"}
        self.assertEqual(data.length, 4)
        self.assertEqual(data.state.params, refstate)

    def test_state_header_state_name(self):
        d = CSVDataImporter(self.state_header_csv_file)
        data = d.load()
        refstate_name = (
            "extra_1.000000e+00_rate_4.000000e+00_str_my_str_temperature_3.000000e+02"
        )
        self.assertEqual(data.state.name, refstate_name)

    def test_state_header_float_strings_parsed_as_floats(self):
        filename = os.path.join(TEST_REFERENCE_DIR, "state_header_string_floats.csv")
        d = CSVDataImporter(filename)
        data = d.load()
        params = data.state.params
        self.assertIsInstance(
            params["rate"],
            float,
            f"Expected float for 'rate', got {type(params['rate'])}: {params['rate']!r}",
        )
        self.assertAlmostEqual(params["rate"], 4.0)
        self.assertIsInstance(
            params["temperature"],
            float,
            f"Expected float for 'temperature', got {type(params['temperature'])}: {params['temperature']!r}",
        )
        self.assertAlmostEqual(params["temperature"], -273.15)
        self.assertIsInstance(
            params["index"],
            float,
            f"Expected float for 'index', got {type(params['index'])}: {params['index']!r}",
        )
        self.assertAlmostEqual(params["index"], 100.0)

    def test_equal(self):
        d_0 = CSVDataImporter(self.default_csv_file)
        d_1 = CSVDataImporter(self.default_csv_file)
        d_2 = copy(d_0)

        self.assertTrue(d_0 == d_1)
        self.assertTrue(d_0 == d_2)

        d_0_wrong_file = CSVDataImporter(self.headerless_csv_file)
        self.assertFalse(d_0 == d_0_wrong_file)

    def test_eq_with_non_importer_returns_NotImplemented(self):
        d = CSVDataImporter(self.default_csv_file)
        self.assertIs(CSVDataImporter.__eq__(d, "not_an_importer"), NotImplemented)
        self.assertIs(CSVDataImporter.__eq__(d, 42), NotImplemented)
        self.assertIs(CSVDataImporter.__eq__(d, None), NotImplemented)

    def assertDefaultDataCorrect(self, data):
        self.assertListEqual(list(data["U"]), [1, 2, 3, 4])
        self.assertListEqual(list(data["F"]), [21, 42, 63, 84])

    @patch("matcal.core.data_importer.sys.platform", "linux")
    def test_csv_dos_import_raise_DOSFileError(self):
        dos_file = os.path.join(TEST_REFERENCE_DIR, "tga_pmdi_dos.csv")
        data_importer = CSVDataImporter(dos_file)
        with self.assertRaises(DOSFileError):
            data_importer.load()

    def test_csv_has_unsupported_character_raise_InvalidCharacterError(self):
        converted_file = os.path.join(TEST_REFERENCE_DIR, "tga_pmdi_converted.csv")
        data_importer = CSVDataImporter(converted_file)
        with self.assertRaises(InvalidCharacterError):
            data_importer.load()

    def test_line_is_not_comment_empty_string(self):
        # Exercises the empty-line branch in _line_is_not_comment.
        d = CSVDataImporter(self.default_csv_file, comments="#")
        self.assertTrue(d._line_is_not_comment(""))


class BatchDataImporterTest(MatcalUnitTest):

    def setUp(self):
        super().setUp(__file__)

    @classmethod
    def setUpClass(cls) -> None:
        cls.csv_batch_pattern = os.path.join(TEST_REFERENCE_DIR, "exp_data_[0-3].csv")
        cls.csv_batch = glob.glob(cls.csv_batch_pattern)

    def test_batch_loader_with_list_init(self):
        dc = BatchDataImporter(self.csv_batch).batch
        s = dc.states
        self.assertEqual(len(dc), 3)
        self.assertEqual(len(s[list(s.keys())[0]].params), 2)

    def test_batch_load_with_set_precision(self):
        b = BatchDataImporter(self.csv_batch_pattern)
        b.set_state_precision(1)
        dc = b.batch
        s = dc.states
        self.assertEqual(len(dc.state_names), 2)
        self.assertEqual(len(s[list(s.keys())[0]].params), 2)

    def test_set_state_precision_rounds_to_exactly_n_sig_figs(self):
        # set_state_precision(n) must round numeric state parameters to exactly
        # n significant figures when reconciling states across a batch.
        # At precision=1, rate=1.1 and rate=1.2 both round to 1.0 and should
        # be merged into a single state.
        f_a = os.path.join(TEST_REFERENCE_DIR, "precision_1sig_a.csv")
        f_b = os.path.join(TEST_REFERENCE_DIR, "precision_1sig_b.csv")
        b = BatchDataImporter([f_a, f_b])
        b.set_state_precision(1)
        dc = b.batch
        self.assertEqual(
            len(dc.state_names),
            1,
            f"Expected 1 merged state at precision=1, got {dc.state_names}",
        )

    def test_batch_loader_with_reg_expression_init(self):
        dc = BatchDataImporter(self.csv_batch_pattern).batch
        s = dc.states
        self.assertEqual(len(dc), 3)
        self.assertEqual(len(dc.state_names), 3)
        self.assertEqual(len(s[list(s.keys())[0]].params), 2)
        for state in s.values():
            self.assertTrue("str" in state.params)
            self.assertTrue("rate" in state.params)

    def test_batch_load_with_additional_states(self):
        b = BatchDataImporter(self.csv_batch)
        b.set_fixed_state_parameters("fixed_states", P1=5.0, P2=4.0)
        dc = b.batch
        self.assertEqual(len(dc), 3)
        s = dc.states
        self.assertEqual(len(s[list(s.keys())[0]].params), 4)
        for state in s.values():
            self.assertIn("str", state.params)
            self.assertIn("rate", state.params)
            self.assertIn("P1", state.params)
            self.assertIn("P2", state.params)

    def test_batch_load_with_file_type(self):
        # file_type is forwarded to FileData(...) via kwargs
        b = BatchDataImporter(self.csv_batch, file_type="csv")
        dc = b.batch
        self.assertEqual(len(dc), 3)

    def test_batch_load_with_file_type_and_fixed_states(self):
        b = BatchDataImporter(self.csv_batch, file_type="csv")
        b.set_fixed_state_parameters(P1=5.0, P2=4.0)
        dc = b.batch
        self.assertEqual(len(dc), 3)
        s = dc.states
        for state in s.values():
            self.assertIn("P1", state.params)
            self.assertIn("P2", state.params)

    def test_batch_loader_pattern_expands_and_loads_expected_files(self):
        """
        Improved replacement for the previous no-op test_batch_loader_error.
        Verifies the glob expands, and that the batch loads the same files.
        """
        pattern = os.path.join(TEST_REFERENCE_DIR, "exp_data_*.csv")
        expected_files = sorted(glob.glob(pattern))
        self.assertGreater(len(expected_files), 0)

        dc = BatchDataImporter(pattern).batch
        # Compare expected absolute paths to the Data.name fields (which are set to abspath)
        expected_abs = sorted(os.path.abspath(f) for f in expected_files)
        loaded_abs = []
        for state in dc:
            for data in dc[state]:
                loaded_abs.append(data.name)
        loaded_abs = sorted(loaded_abs)

        self.assertListEqual(loaded_abs, expected_abs)

    def test_batch_loader_pattern_matches_no_files_raises(self):
        pattern = os.path.join(TEST_REFERENCE_DIR, "this_matches_nothing_*.csv")
        with self.assertRaises(FileNotFoundError):
            BatchDataImporter(pattern)

    def test_batch_loader_invalid_filenames_type_raises(self):
        with self.assertRaises(TypeError):
            BatchDataImporter(123)

        with self.assertRaises(TypeError):
            BatchDataImporter({"a": "b"})

    def test_set_fixed_state_parameters_name_type_validation(self):
        b = BatchDataImporter(self.csv_batch_pattern)
        with self.assertRaises(TypeError):
            b.set_fixed_state_parameters(name=123, P1=1.0)

    def test_set_fixed_state_parameters_param_type_validation(self):
        b = BatchDataImporter(self.csv_batch_pattern)
        with self.assertRaises(TypeError):
            b.set_fixed_state_parameters(P1=None)
        with self.assertRaises(TypeError):
            b.set_fixed_state_parameters(P1=SolitaryState())

    def test_set_fixed_state_parameters_requires_string_or_real_values(self):
        b = BatchDataImporter(self.csv_batch_pattern)
        b.set_fixed_state_parameters(P1=5.0, P2="abc")  # ok

        with self.assertRaises(TypeError):
            b.set_fixed_state_parameters(P1=[1, 2, 3])  # not allowed

    def test_fixed_state_name_used_only_when_no_file_states(self):
        # Use comment files that have no state header => SolitaryState() in both
        files = [
            os.path.join(TEST_REFERENCE_DIR, "comment_header.csv"),
            os.path.join(TEST_REFERENCE_DIR, "comment_header_and_tail.csv"),
        ]
        b = BatchDataImporter(files, comments="$")
        b.set_fixed_state_parameters(name="my_fixed_state", P1=1.0)

        dc = b.batch
        self.assertEqual(len(dc.state_names), 1)
        self.assertEqual(dc.state_names[0], "my_fixed_state")
        s = dc.states["my_fixed_state"]
        self.assertIn("P1", s.params)
        self.assertEqual(s.params["P1"], 1.0)

    def test_fixed_state_name_ignored_when_file_states_exist(self):
        # Use batch files that include states in the headers
        b = BatchDataImporter(self.csv_batch_pattern)
        b.set_fixed_state_parameters(name="my_fixed_state", P1=1.0)

        # We cannot easily assert on logs here without a capture helper,
        # but we can assert the name is not adopted.
        dc = b.batch
        self.assertNotIn("my_fixed_state", dc.state_names)

    def test_batch_loader_forwards_comments_kwarg(self):
        """
        Verify BatchDataImporter forwards kwargs into FileData/CSVDataImporter by using
        files that require comments="$".
        """
        files = [
            os.path.join(TEST_REFERENCE_DIR, "comment_header.csv"),
            os.path.join(TEST_REFERENCE_DIR, "comment_header_and_tail.csv"),
        ]

        # Without forwarding comments="$", parsing should fail (comment lines not skipped)
        with self.assertRaises(Exception):
            BatchDataImporter(files).batch

        dc = BatchDataImporter(files, comments="$").batch
        self.assertEqual(len(dc[dc.state_names[0]]), 2)
        for d in dc[dc.state_names[0]]:
            self.assertIn("time", d.field_names)
            self.assertIn("temp", d.field_names)
            self.assert_close_arrays([0, 1, 2], d["time"])
            self.assert_close_arrays([100, 200, 300], d["temp"])

    def test_batch_loader_mismatched_state_variables_raises(self):
        """
        Create two temporary CSVs with different state variable keys and verify the
        reconciliation step throws BatchDataImporterStateError.
        """
        f1 = "batch_state_a.csv"
        f2 = "batch_state_b.csv"

        s1 = """{"temperature": 300}
# comment
X,Y
0,0
1,1
"""
        s2 = """{"rate": 1.0}
# comment
X,Y
0,0
1,1
"""
        with open(f1, "w") as fh:
            fh.write(s1)
        with open(f2, "w") as fh:
            fh.write(s2)

        b = BatchDataImporter([f1, f2], comments="#", file_type="csv")
        with self.assertRaises(BatchDataImporter.BatchDataImporterStateError):
            _ = b.batch

    def test_empty_list_raises_ValueError(self):
        with self.assertRaises(ValueError):
            BatchDataImporter([])

    def test_set_state_precision_zero_raises_ValueError(self):
        # precision=0 causes set_significant_figures to return 0.0 for all
        # values, collapsing every numeric state into one. This is never the
        # intended behaviour. set_state_precision must reject precision < 1.
        b = BatchDataImporter(self.csv_batch_pattern)
        with self.assertRaises(ValueError):
            b.set_state_precision(0)

    def test_states_property_returns_states_from_batch(self):
        b = BatchDataImporter(self.csv_batch_pattern)
        states = b.states
        self.assertGreater(len(list(states.keys())), 0)

    def test_filenames_property_returns_matched_files(self):
        b = BatchDataImporter(self.csv_batch_pattern)
        self.assertEqual(sorted(b.filenames), sorted(self.csv_batch))

    def test_set_fixed_state_parameters_non_string_key_raises_type_error(self):
        b = BatchDataImporter(self.csv_batch_pattern)
        with self.assertRaises(TypeError):
            b.set_fixed_state_parameters(**{123: 1.0})


class FileEncodingTest(MatcalUnitTest):

    def setUp(self):
        super().setUp(__file__)

    def test_has_dos_newlines_returns_false_for_lf_file(self):
        filename = "lf_test.csv"
        with open(filename, "wb") as f:
            f.write(b"time,temp\n0,100\n1,200\n")

        self.assertFalse(_has_dos_newlines(filename))

    def test_has_dos_newlines_detects_crlf_across_chunk_boundary(self):
        filename = "chunk_boundary_crlf_test.csv"

        # With chunk_size=4, the first chunk is b"abc\r" and the next starts
        # with b"\n", so this verifies that _has_dos_newlines correctly handles
        # CRLF split across chunk boundaries.
        with open(filename, "wb") as f:
            f.write(b"abc\r\ndef\n")

        self.assertTrue(_has_dos_newlines(filename, chunk_size=4))

    @patch("matcal.core.data_importer.sys.platform", "linux")
    def test_is_dos_returns_true_for_crlf_on_non_windows(self):
        filename = "linux_crlf_test.csv"
        with open(filename, "wb") as f:
            f.write(b"time,temp\r\n0,100\r\n")

        self.assertTrue(_is_dos(filename))

    @patch("matcal.core.data_importer.sys.platform", "linux")
    def test_is_dos_returns_false_for_lf_on_non_windows(self):
        filename = "linux_lf_test.csv"
        with open(filename, "wb") as f:
            f.write(b"time,temp\n0,100\n")

        self.assertFalse(_is_dos(filename))

    @patch("matcal.core.data_importer.sys.platform", "win32")
    def test_is_dos_returns_false_on_windows_even_for_crlf(self):
        filename = "windows_crlf_test.csv"
        with open(filename, "wb") as f:
            f.write(b"time,temp\r\n0,100\r\n")

        self.assertFalse(_is_dos(filename))

    def test_has_invalid_lines_returns_false_for_empty_string(self):
        self.assertFalse(_has_invalid_lines(""))

    def test_has_invalid_lines_returns_true_for_nonempty_string(self):
        self.assertTrue(_has_invalid_lines("2 min,\xb0C,%,%/min\n"))

    def test_has_invalid_utc_characters_returns_string(self):
        converted_file = os.path.join(TEST_REFERENCE_DIR, "tga_pmdi_converted.csv")
        goal = "2 min,\xb0C,%,%/min\n"
        self.assertEqual(_report_invalid_utc_lines(converted_file), goal)


class NumpyDataImporterTest(MatcalUnitTest):

    def setUp(self):
        super().setUp(__file__)
        self._example_data = convert_dictionary_to_data(
            {"A": [1, 2, 3], "Temp": np.array([100, 200, 300])}
        )
        self._example_file = "unittest.npy"

        np.save(self._example_file, self._example_data)

    def test_init(self):
        np_data_importer = NumpyDataImporter(self._example_file)
        self.assertEqual(np_data_importer.filename, self._example_file)

    def test_load_npy(self):
        np_data_importer = NumpyDataImporter(self._example_file)
        data = np_data_importer.load()
        self.assertTrue(data.field_names == ["A", "Temp"])
        self.assertIsInstance(data, Data)
        self.assertTrue(np.allclose(self._example_data["A"], data["A"]))
        self.assertTrue(np.allclose(self._example_data["Temp"], data["Temp"]))
        self.assertEqual(data.name, os.path.abspath(self._example_file))

    def test_empty(self):
        # Define the structured data type
        dtype = np.dtype([("x", np.float64), ("Y", np.float64)])

        # Create an empty structured array
        empty_array = np.empty(0, dtype=dtype)
        np.save("test.npy", empty_array)
        np_data_importer = NumpyDataImporter("test.npy")
        with self.assertRaises(ValueError):
            data = np_data_importer.load()


class DataImporterFactoryTest(MatcalUnitTest):

    def setUp(self):
        super().setUp(__file__)
        self.default_csv_file = os.path.join(TEST_REFERENCE_DIR, "default.csv")

    @classmethod
    def setUpClass(cls) -> None:
        cls.state_header_csv_file_user_ext = os.path.join(
            TEST_REFERENCE_DIR, "state_header.user_ext"
        )
        cls.example_state = State("example")

    def test_non_existent_file_will_throw_file_not_found_error(self):
        with self.assertRaises(FileNotFoundError):
            d_0 = FileData("invalid_file.csv")

    def test_invalid_key_error(self):
        with self.assertRaises(KeyError):
            d_0 = FileData("invalid_key.key")

    def test_file_without_extension_raises_ValueError(self):
        with open("no_extension", "w") as f:
            f.write("U,F\n1,2\n")
        with self.assertRaises(ValueError):
            FileData("no_extension")

    def test_invalid_file_type_type_error(self):
        with self.assertRaises(TypeError):
            d_0 = FileData("invalid_key.key", file_type=1)

    def test_valid_default_file(self):
        data = FileData(self.default_csv_file)
        self.assert_default_data_correct(data)
        self.assertTrue(data.state == SolitaryState())

    def assert_default_data_correct(self, data):
        self.assertListEqual(list(data["U"]), [1, 2, 3, 4])
        self.assertListEqual(list(data["F"]), [21, 42, 63, 84])

    def test_read_file_change_state(self):
        data = FileData(self.default_csv_file, state=self.example_state)
        self.assert_default_data_correct(data)
        self.assertTrue(data.state == self.example_state)

    def test_read_file_user_file_extension(self):
        data = FileData(self.state_header_csv_file_user_ext, file_type="csv")
        self.assert_default_data_correct(data)
        refstate = {"rate": 4.0, "temperature": 300.0, "extra": 1.0, "str": "my_str"}
        self.assertTrue(data.state.params == refstate)

    def test_read_numpy_file(self):
        example_data = convert_dictionary_to_data(
            {"A": [1, 2, 3], "Temp": np.array([100, 200, 300])}
        )
        example_file = "unittest.npy"
        np.save(example_file, example_data)

        data = FileData(example_file)

        self.assertTrue(np.allclose(example_data["A"], data["A"]))
        self.assertTrue(np.allclose(example_data["Temp"], data["Temp"]))

    def test_drop_NaNs(self):
        with self.assertRaises(ValueError):
            data = FileData(os.path.join(TEST_REFERENCE_DIR, "data_with_nans_infs.csv"))

        data = FileData(
            os.path.join(TEST_REFERENCE_DIR, "data_with_nans_infs.csv"), drop_NaNs=True
        )

        data_gold = np.genfromtxt(
            os.path.join(TEST_REFERENCE_DIR, "data_with_nans_infs.csv"),
            skip_header=1,
            delimiter=",",
        )
        data_gold = data_gold[np.isfinite(data_gold).all(axis=1), :]

        self.assert_close_arrays(data_gold[:, 0], data["load"])
        self.assert_close_arrays(data_gold[:, 1], data["displacement"])


class TestCSVFileData(MatcalUnitTest):
    def setUp(self) -> None:
        super().setUp(__file__)
        self._field_data_snapshot_file = os.path.join(
            TEST_REFERENCE_DIR, "17_A-4B_pull-sys1-0000_0.csv"
        )
        self.field_data = FileData(self._field_data_snapshot_file)
        self.ref_data_file = os.path.join(TEST_REFERENCE_DIR, "x_csv_array.csv")
        self.ref_data = read_csv(self.ref_data_file, ["X"])

    def test_get_data_dataframe(self):
        self.assertIsInstance(self.field_data, Data)

    def test_get_X_position_information(self):
        self.assertIsInstance(self.field_data["X"], Data)

    def test_confirm_keys(self):
        base_keys = [
            "X",
            "Y",
            "Z",
            "U",
            "V",
            "W",
            "exx",
            "eyy",
            "exy",
            "e1",
            "e2",
            "gamma",
            "sigma",
            "x",
            "y",
            "u",
            "v",
            "q",
            "r",
            "q_ref",
            "r_ref",
        ]
        data_frame_keys = self.field_data.field_names
        for key in base_keys:
            self.assertIn(key, data_frame_keys)

    def test_check_one_data_array(self):
        x_array = self.field_data["X"]
        ref_x_array = self.ref_data["X"]
        self.assert_close_arrays(x_array, ref_x_array)

    def test_skip_commented_tail_line(self):
        filename = os.path.join(TEST_REFERENCE_DIR, "comment_trailing.csv")
        self._confirm_matching_data(filename)

    def test_skip_commented_header_and_tail_line(self):
        filename = os.path.join(TEST_REFERENCE_DIR, "comment_header_and_tail.csv")
        self._confirm_matching_data(filename)

    def test_skip_commented_middle_line(self):
        filename = os.path.join(TEST_REFERENCE_DIR, "comment_middle.csv")
        self._confirm_matching_data(filename)

    def test_skip_commented_header_line(self):
        filename = os.path.join(TEST_REFERENCE_DIR, "comment_header.csv")
        self._confirm_matching_data(filename)

    def test_skip_commented_header_line_with_state(self):
        filename = os.path.join(TEST_REFERENCE_DIR, "comment_header_with_state.csv")
        data = self._confirm_matching_data(filename)
        state_params = data.state.params
        state_gold = {"state_param_1": 1, "state_param_2": "state_string"}
        self.assertEqual(state_params, state_gold)

    def test_skip_commented_dispersed_with_state(self):
        filename = os.path.join(TEST_REFERENCE_DIR, "comments_dispersed_with_state.csv")
        data = self._confirm_matching_data(filename)
        state_params = data.state.params
        state_gold = {
            "state_param_1": "state_string1",
            "state_param_2": 1,
            "state_param_3": "state_string2",
        }

        self.assertEqual(state_params, state_gold)
        self.assertEqual(
            data.state.name,
            (
                "state_param_1_state_string1"
                f"_state_param_2_{1:12.6e}_state_param_3_state_string2"
            ),
        )

    def test_skip_commented_header_line_multichar_prefix(self):
        filename = os.path.join(TEST_REFERENCE_DIR, "comment_header_multichar.csv")
        data = FileData(filename, comments="##")
        goal_dict = {"time": [0, 1, 2], "temp": [100, 200, 300]}
        for goal_key, goal_value in goal_dict.items():
            self.assertIn(goal_key, data.field_names)
            self.assert_close_arrays(goal_value, data[goal_key])

    def test_skip_comments_passed_as_list_raises_TypeError(self):
        # comments must be a single string. Passing a list raises TypeError
        # because neither genfromtxt nor _line_is_not_comment support lists.
        filename = os.path.join(TEST_REFERENCE_DIR, "comment_header.csv")
        with self.assertRaises(TypeError):
            FileData(filename, comments=["$"])

    def test_error_with_incorrect_state_dict(self):
        file_string = """{temperature:100}
        # a comment
        # another
        load, displacement
        1.0, 0.0
        2.0,1.0
        """
        with open("test_file.csv", "w") as f:
            f.write(file_string)
        with self.assertRaises((ValueError, SyntaxError)):
            data = FileData("test_file.csv", comments="#")

    def test_malformed_state_header_raises_ValueError_with_filename(self):
        # When ast.literal_eval fails on a dict-like header line, the raised
        # ValueError must include the filename so the user can identify the
        # offending file.
        with open("bad_state_header.csv", "w") as f:
            f.write("{temperature:100}\n")
            f.write("load,displacement\n")
            f.write("1.0,2.0\n")
        with self.assertRaises(ValueError) as ctx:
            FileData("bad_state_header.csv")
        self.assertIn("bad_state_header.csv", str(ctx.exception))

    def _confirm_matching_data(self, filename):
        data = FileData(filename, comments="$")
        goal_dict = {"time": [0, 1, 2], "temp": [100, 200, 300]}
        for goal_key, goal_value in goal_dict.items():
            self.assertIn(goal_key, data.field_names)
            self.assert_close_arrays(goal_value, data[goal_key])
        return data


class TestMatlabFileDataMatV7(MatcalUnitTest):
    def setUp(self) -> None:
        super().setUp(__file__)
        self._dic_snapshot_file = os.path.join(TEST_REFERENCE_DIR, "simple_2D_dic.mat")
        self.dic_data = FileData(self._dic_snapshot_file)

    def test_get_data_dataframe(self):
        self.assertIsInstance(self.dic_data, Data)

    def test_get_X_position_information(self):
        self.assertIsInstance(self.dic_data["X"], Data)

    def test_confirm_keys(self):
        base_keys = ["X", "Y", "U_x", "U_y", "T", "E"]
        data_frame_keys = self.dic_data.field_names
        for key in base_keys:
            self.assertIn(key, data_frame_keys)

    def test_check_one_data_array(self):
        x_array = self.dic_data["E"]
        ref_x_array = 100 * np.ones(20 * 20)
        self.assertTrue(np.allclose(x_array, ref_x_array))
