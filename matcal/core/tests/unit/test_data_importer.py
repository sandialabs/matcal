import os
import glob
import numpy as np
from copy import copy
import csv
import unittest

from matcal.core.data_importer import (CSVDataImporter, DOSFileError, FileData,
                                        InvalidCharacterError, _get_unix_file_report, 
                                        _report_invalid_utc_lines)
from matcal.core.data_importer import NumpyDataImporter, BatchDataImporter
from matcal.core.data import convert_dictionary_to_data, Data
from matcal.core.state import State, SolitaryState
from matcal.core.tests.MatcalUnitTest import MatcalUnitTest, is_mac


TEST_RERERENCE_DIR = os.path.join(os.path.dirname(__file__), 
                                  "test_reference", "data_importer")

def read_csv(filename, header=None):
    with open(filename, 'r') as csvfile:

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
        cls.default_csv_file = os.path.join(TEST_RERERENCE_DIR, "default.csv")
        cls.headerless_csv_file = os.path.join(TEST_RERERENCE_DIR, "headerless.csv")
        cls.state_header_csv_file = os.path.join(TEST_RERERENCE_DIR, "state_header.csv")
        cls.example_state = State("example")
        cls.csv_batch_pattern = os.path.join(TEST_RERERENCE_DIR, "exp_data_[0-3].csv")
        
        cls.csv_batch = glob.glob(cls.csv_batch_pattern)
        cls.dashed_header_file = os.path.join(TEST_RERERENCE_DIR, "data_with_dashes.csv")

    def test_non_existent_file_will_throw_file_not_found_error(self):
        with self.assertRaises(FileNotFoundError):
            d_0 = CSVDataImporter("invalid_file.csv")

    def test_invalid_filename_type(self):
        with self.assertRaises(TypeError):
            d0 = CSVDataImporter(1)

    def test_import_header_with_dashes(self):
        d_imp = CSVDataImporter(self.dashed_header_file)
        data = d_imp.load()
        goal_fields = ['time', 'field-A', 'field-1', 'field-a']
        test_fields = data.field_names
        self.assertEqual(len(goal_fields), len(test_fields))
        for goal in goal_fields:
            self.assertIn(goal, test_fields)

    def test_non_interpretable_file_will_throw_data_format_error(self):
        with self.assertRaises(TypeError):
            d_nan = CSVDataImporter(os.path.join(TEST_RERERENCE_DIR, "nan_test.csv"), 
                                import_strings=True)
            d_nan.load()

        with self.assertRaises(TypeError):
            d_inf = CSVDataImporter(os.path.join(TEST_RERERENCE_DIR, "inf_test.csv"))
            d_inf.load()

        with self.assertRaises(TypeError):
            d_str = CSVDataImporter(os.path.join(TEST_RERERENCE_DIR, "str_test.csv"))
            d_str.load()

        with self.assertRaises(TypeError):
            d_char = CSVDataImporter(os.path.join(TEST_RERERENCE_DIR, "char_test.csv"))
            d_char.load()

        with self.assertRaises(TypeError):
            d_test = CSVDataImporter(os.path.join(TEST_RERERENCE_DIR, "non_interpretable_data.csv"))
            d_test.load()

    def test_drop_NaNs(self):
        with self.assertRaises(TypeError):
            d_str = CSVDataImporter(os.path.join(TEST_RERERENCE_DIR, "data_with_nans_infs.csv"))
            data = d_str.load()

        d_str = CSVDataImporter(os.path.join(TEST_RERERENCE_DIR, "data_with_nans_infs.csv"),
                                 drop_NaNs=True)
        data = d_str.load()

        data_gold = np.genfromtxt(os.path.join(TEST_RERERENCE_DIR, "data_with_nans_infs.csv")
                                  , skip_header=1, delimiter=",")
        data_gold = data_gold[np.isfinite(data_gold).all(axis=1), :]
 
        self.assert_close_arrays(data_gold[:,0], data["load"])
        self.assert_close_arrays(data_gold[:,1], data["displacement"])
        
    def test_read_strings(self):
        d_str = CSVDataImporter(os.path.join(TEST_RERERENCE_DIR, "str_test.csv"), 
                                import_strings=True)
        data = d_str.load()
        self.assertTrue("e" in data["load"])

        d_char = CSVDataImporter(os.path.join(TEST_RERERENCE_DIR, "char_test.csv"),
                                 import_strings=True)
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
            data = d0.load()

    def test_empty_column(self):
        with open("empty.csv", "w") as f:
            f.write("X, Y, Z\n")
            f.write("0, , 0\n")
            f.write("0, , 0\n")

        d0 = CSVDataImporter("empty.csv")
        with self.assertRaises(TypeError):
            data = d0.load()

    def test_field_names(self):
        d_0 = CSVDataImporter(self.default_csv_file)
        data = d_0.load()
        self.assertListEqual(list(data.field_names), ["U", "F"])

    def test_get_length_of_data(self):
        D = CSVDataImporter(self.default_csv_file)
        data = D.load()
        self.assertEqual(data.length, 4)

    def test_load_data(self):
        d_0 = CSVDataImporter(self.default_csv_file)
        data = d_0.load()
        import sys
        self.assertTrue(sys.getsizeof(data) > 50)

    def test_state_header_values(self):
        D = CSVDataImporter(self.state_header_csv_file)
        data = D.load()
        refstate = {'rate': 4.0, 'temperature': 300, 'extra': 1, "str":"my_str"}
        self.assertEqual(data.length, 4)
        self.assertEqual(data.state.params, refstate)

    def test_state_header_state_name(self):
        D = CSVDataImporter(self.state_header_csv_file)
        data = D.load()
        refstate_name = "extra_1.000000e+00_rate_4.000000e+00_str_my_str_temperature_3.000000e+02"
        self.assertEqual(data.state.name, refstate_name)

    def test_batch_loader_with_list_init(self):
        dc = BatchDataImporter(self.csv_batch).batch
        s = dc.states
        self.assertEqual(len(dc), 3)
        self.assertEqual(len(s[list(s.keys())[0]].params), 2)

    def test_batch_load_with_set_precision(self):
        b = BatchDataImporter(self.csv_batch_pattern)
        b.set_options(state_precision=0)
        dc = b.batch
        s = dc.states
        self.assertEqual(len(dc.state_names), 2)
        self.assertEqual(len(s[list(s.keys())[0]].params), 2)

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
        b = BatchDataImporter(self.csv_batch, fixed_states={"P1": 5.0, "P2": 4.0})
        dc = b.batch
        self.assertEqual(len(dc), 3)
        s = dc.states
        self.assertEqual(len(s[list(s.keys())[0]].params), 4)
        for state in s.values():
            self.assertTrue("str" in state.params)
            self.assertTrue("rate" in state.params)
            self.assertTrue("P1" in state.params)
            self.assertTrue("P2" in state.params)

    def test_batch_load_with_file_type(self):
        b = BatchDataImporter(self.csv_batch, file_type="csv")
        dc = b.batch
        self.assertEqual(len(dc), 3)
        s = dc.states
        self.assertEqual(len(s[list(s.keys())[0]].params), 2)
        for state in s.values():
            self.assertTrue("str" in state.params)
            self.assertTrue("rate" in state.params)

    def test_batch_load_with_file_type_and_fixed_states(self):
        b = BatchDataImporter(self.csv_batch, file_type="csv", fixed_states={"P1": 5.0, "P2": 4.0})
        dc = b.batch
        self.assertEqual(len(dc), 3)
        s = dc.states
        self.assertEqual(len(s[list(s.keys())[0]].params), 4)
        for state in s.values():
            self.assertTrue("str" in state.params)
            self.assertTrue("rate" in state.params)
            self.assertTrue("P1" in state.params)
            self.assertTrue("P2" in state.params)

    def test_batch_loader_error(self):
        pattern = os.path.join(TEST_RERERENCE_DIR, "exp_data_*.csv")
        B = BatchDataImporter(pattern).batch

    def test_equal(self):
        d_0 = CSVDataImporter(self.default_csv_file)
        d_1 = CSVDataImporter(self.default_csv_file)

        d_2 = copy(d_0)

        self.assertTrue(d_0 == d_1)
        self.assertTrue(d_0 == d_2)

        d_0_wrong_file = CSVDataImporter(self.headerless_csv_file)
        self.assertFalse(d_0 == d_0_wrong_file)

    def assertDefaultDataCorrect(self, data):
        self.assertListEqual(list(data['U']), [1, 2, 3, 4])
        self.assertListEqual(list(data['F']), [21, 42, 63, 84])

    def test_csv_dos_import_raise_DOSFileError(self):
        dos_file = os.path.join(TEST_RERERENCE_DIR, "tga_pmdi_dos.csv")
        data_importer = CSVDataImporter(dos_file)
        with self.assertRaises(DOSFileError):
            data = data_importer.load()
        
    def test_csv_has_unsupported_character_raise_InvalidCharacterError(self):
        converted_file = os.path.join(TEST_RERERENCE_DIR, "tga_pmdi_converted.csv")
        data_importer = CSVDataImporter(converted_file)
        with self.assertRaises(InvalidCharacterError):
            data = data_importer.load()

    
class FileEncodingTest(MatcalUnitTest):

    def setUp(self):
        super().setUp(__file__)

    @unittest.skipIf(is_mac(), "mac defaults do not work for this")
    def test_return_dos_report(self):
        dos_file = os.path.join(TEST_RERERENCE_DIR, "tga_pmdi_dos.csv")
        goal = ["ISO-8859 text, with CRLF line terminators", "CSV text"]     
        self.assertTrue(_get_unix_file_report(dos_file) in goal)

    @unittest.skipIf(is_mac(), "mac defaults do not work for this")
    def test_return_converted_dos_report(self):
        converted_file = os.path.join(TEST_RERERENCE_DIR, "tga_pmdi_converted.csv")
        goal = ["ISO-8859 text", "CSV text"]     
        self.assertTrue(_get_unix_file_report(converted_file) in goal)

    @unittest.skipIf(is_mac(), "mac defaults do not work for this")
    def test_return_converted_and_cleaned_dos_report(self):
        unix_file = os.path.join(TEST_RERERENCE_DIR, "tga_pmdi_unix.csv")
        goal = ["ASCII text", "CSV text"]     
        self.assertTrue(_get_unix_file_report(unix_file) in goal)
    
    @unittest.skipIf(is_mac(), "mac defaults do not work for this")
    def test_return_ascii_report(self):
        ascii_file = os.path.join(TEST_RERERENCE_DIR, "x_array.csv")
        goal = "ASCII text"        
        self.assertEqual(_get_unix_file_report(ascii_file), goal)

    @unittest.skipIf(is_mac(), "mac defaults do not work for this")
    def test_has_invalid_utc_characters_returns_string(self):
        converted_file = os.path.join(TEST_RERERENCE_DIR, "tga_pmdi_converted.csv")
        goal = '2 min,\xb0C,%,%/min\n'       
        self.assertEqual(_report_invalid_utc_lines(converted_file), goal)


class NumpyDataImporterTest(MatcalUnitTest):

    def setUp(self):
        super().setUp(__file__)
        self._example_data = convert_dictionary_to_data({"A": [1, 2, 3], "Temp": np.array([100, 200, 300])})
        self._example_file = "unittest.npy"

        np.save(self._example_file, self._example_data)

    def test_init(self):
        np_data_importer = NumpyDataImporter(self._example_file)

    def test_load_npy(self):
        np_data_importer = NumpyDataImporter(self._example_file)
        data = np_data_importer.load()
        self.assertTrue(data.field_names == ["A", "Temp"])
        self.assertIsInstance(data, Data)
        self.assertTrue(np.allclose(self._example_data["A"], data["A"]))
        self.assertTrue(np.allclose(self._example_data["Temp"], data["Temp"]))
        self.assertTrue(data.name, os.path.abspath(self._example_file))

    def test_empty(self):
        # Define the structured data type
        dtype = np.dtype([('x', np.float64), ('Y', np.float64)])

        # Create an empty structured array
        empty_array = np.empty(0, dtype=dtype)
        np.save("test.npy", empty_array)
        np_data_importer = NumpyDataImporter("test.npy")
        with self.assertRaises(ValueError):
            data = np_data_importer.load()


class DataImporterFactoryTest(MatcalUnitTest):

    def setUp(self):
        super().setUp(__file__)
        self.default_csv_file = os.path.join(TEST_RERERENCE_DIR, "default.csv")

    @classmethod
    def setUpClass(cls) -> None:
        cls.state_header_csv_file_user_ext = os.path.join(TEST_RERERENCE_DIR, 
                                                          "state_header.user_ext")
        cls.example_state = State("example")

    def test_non_existent_file_will_throw_file_not_found_error(self):
        with self.assertRaises(FileNotFoundError):
            d_0 = FileData("invalid_file.csv")

    def test_invalid_key_error(self):
        with self.assertRaises(KeyError):
            d_0 = FileData("invalid_key.key")

    def test_invalid_file_type_type_error(self):
        with self.assertRaises(TypeError):
            d_0 = FileData("invalid_key.key", file_type=1)

    def test_valid_default_file(self):
        data = FileData(self.default_csv_file)
        self.assert_default_data_correct(data)
        self.assertTrue(data.state == SolitaryState())

    def assert_default_data_correct(self, data):
        self.assertListEqual(list(data['U']), [1, 2, 3, 4])
        self.assertListEqual(list(data['F']), [21, 42, 63, 84])

    def test_read_file_change_state(self):
        data = FileData(self.default_csv_file, state=self.example_state)
        self.assert_default_data_correct(data)
        self.assertTrue(data.state == self.example_state)

    def test_read_file_user_file_extension(self):
        data = FileData(self.state_header_csv_file_user_ext, file_type="csv")
        self.assert_default_data_correct(data)
        refstate = {'rate': 4.0, 'temperature': 300, 'extra': 1, "str":"my_str"}
        self.assertTrue(data.state.params == refstate)

    def test_read_numpy_file(self):
        example_data = convert_dictionary_to_data({"A": [1, 2, 3], "Temp": np.array([100, 200, 300])})
        example_file = "unittest.npy"
        np.save(example_file, example_data)

        data = FileData(example_file)

        self.assertTrue(np.allclose(example_data["A"], data["A"]))
        self.assertTrue(np.allclose(example_data["Temp"], data["Temp"]))

    def test_drop_NaNs(self):
        with self.assertRaises(TypeError):
            data = FileData(os.path.join(TEST_RERERENCE_DIR, "data_with_nans_infs.csv"))

        data = FileData(os.path.join(TEST_RERERENCE_DIR, "data_with_nans_infs.csv"),
                         drop_NaNs=True)

        data_gold = np.genfromtxt(os.path.join(TEST_RERERENCE_DIR, "data_with_nans_infs.csv"),
                                   skip_header=1, delimiter=",")
        data_gold = data_gold[np.isfinite(data_gold).all(axis=1), :]
 
        self.assert_close_arrays(data_gold[:,0], data["load"])
        self.assert_close_arrays(data_gold[:,1], data["displacement"])
        

class TestCSVFileData(MatcalUnitTest):
    def setUp(self) -> None:
        super().setUp(__file__)
        self._field_data_snapshot_file = os.path.join(TEST_RERERENCE_DIR, 
                                                      "17_A-4B_pull-sys1-0000_0.csv")
        self.field_data = FileData(self._field_data_snapshot_file)
        self.ref_data_file = os.path.join(TEST_RERERENCE_DIR, "x_csv_array.csv")
        self.ref_data = read_csv(self.ref_data_file, ["X"])

    def test_get_data_dataframe(self):
        self.assertIsInstance(self.field_data, Data)

    def test_get_X_position_information(self):
        self.assertIsInstance(self.field_data["X"], Data)

    def test_confirm_keys(self):
        base_keys = ["X", "Y", "Z", "U", "V", "W",
                     "exx", "eyy", "exy", "e1", "e2",
                     "gamma", "sigma", "x", "y", "u",
                     "v", "q", "r", "q_ref", "r_ref"]
        data_frame_keys = self.field_data.field_names
        for key in base_keys:
            self.assertIn(key, data_frame_keys)

    def test_check_one_data_array(self):
        x_array = self.field_data["X"]
        ref_x_array = self.ref_data["X"]
        self.assert_close_arrays(x_array, ref_x_array)

    def test_skip_commented_tail_line(self):
        filename = os.path.join(TEST_RERERENCE_DIR, "comment_trailing.csv")
        self._confirm_matching_data(filename)

    def test_skip_commented_header_and_tail_line(self):
        filename = os.path.join(TEST_RERERENCE_DIR, "comment_header_and_tail.csv")
        self._confirm_matching_data(filename)

    def test_skip_commented_middle_line(self):
        filename = os.path.join(TEST_RERERENCE_DIR, "comment_middle.csv")
        self._confirm_matching_data(filename)

    def test_skip_commented_header_line(self):
        filename = os.path.join(TEST_RERERENCE_DIR, "comment_header.csv")
        self._confirm_matching_data(filename)

    def test_skip_commented_header_line_with_state(self):
        filename = os.path.join(TEST_RERERENCE_DIR, "comment_header_with_state.csv")
        data = self._confirm_matching_data(filename)
        state_params = data.state.params
        state_gold = {"state_param_1":1, "state_param_2":"state_string"}
        self.assertEqual(state_params, state_gold)

    def test_skip_commented_disperesed_with_state(self):
        filename = os.path.join(TEST_RERERENCE_DIR, "comments_dispersed_with_state.csv")
        data = self._confirm_matching_data(filename)
        state_params = data.state.params
        state_gold = {"state_param_1":"state_string1", "state_param_2":1, 
                      "state_param_3":"state_string2"}

        self.assertEqual(state_params, state_gold)
        self.assertEqual(data.state.name, 
                         ("state_param_1_state_string1"
                          f"_state_param_2_{1:12.6e}_state_param_3_state_string2"))
    def _confirm_matching_data(self, filename):
        data = FileData(filename, comments="$")
        goal_dict = {'time': [0, 1, 2], 'temp': [100, 200, 300]}
        for goal_key, goal_value in goal_dict.items():
            self.assertIn(goal_key, data.field_names)
            self.assert_close_arrays(goal_value, data[goal_key])
        return data

class TestMatlabFileDataMatV7(MatcalUnitTest):
    def setUp(self) -> None:
        super().setUp(__file__)
        self._dic_snapshot_file = os.path.join(TEST_RERERENCE_DIR, "simple_2D_dic.mat")
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
