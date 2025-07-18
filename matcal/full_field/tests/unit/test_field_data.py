from matcal.core.tests.MatcalUnitTest import MatcalUnitTest
from matcal.core.data import Data
from matcal.full_field.data_importer import CSVFieldDataSeriesParser
import numpy as np


class TestImportCSVFieldDataSeriesImporter(MatcalUnitTest):
    def setUp(self) -> None:
        super().setUp(__file__)
        self._field_data_global_data_file = self.get_current_files_path(__file__) + \
                                            "/input_files/csv_global_data.csv"
        self._field_data_series_directory = self.get_current_files_path(__file__) + "/input_files/csv_data_series"
        self.field_data = CSVFieldDataSeriesParser(self._field_data_global_data_file,
                                                     self._field_data_series_directory)

    def test_raise_error_for_bad_global_file(self):
        self.assert_error_type(CSVFieldDataSeriesParser.FieldDataDataSeriesMissingPathObject, CSVFieldDataSeriesParser,
            "not_a_file", self._field_data_series_directory)

    def test_raise_error_for_bad_directory(self):
        self.assert_error_type(CSVFieldDataSeriesParser.FieldDataDataSeriesMissingPathObject, CSVFieldDataSeriesParser,
                               self._field_data_global_data_file, "not_a_dir")

    def test_get_number_of_frames(self):
        goal_number_frames = 3
        self.assertEqual(self.field_data.number_of_frames, goal_number_frames)

    def test_return_first_field_data_frame(self):
        data = self.field_data.get_frame(0)
        self.assertIsInstance(data, Data)
        self.assertEqual(data.name.split('/')[-1], "test_0.csv")

    def test_return_last_field_data_frame(self):
        data = self.field_data.get_frame(2)
        self.assertIsInstance(data, Data)
        self.assertEqual(data.name.split('/')[-1], "test_2.csv")

    def test_raise_error_for_bad_frame_index(self):
        self.assert_error_type(self.field_data.FieldDataDataSeriesBadFrameIndex, self.field_data.get_frame, 10)
        self.assert_error_type(self.field_data.FieldDataDataSeriesBadFrameIndex, self.field_data.get_frame, -1)
        self.assert_error_type(self.field_data.FieldDataDataSeriesBadFrameIndex, self.field_data.get_frame, None)

    def test_return_time_at_first_frame(self):
        time = self.field_data.get_global_data()['time'][0]
        self.assertAlmostEqual(time, 0)

    def test_return_time_at_last_frame(self):
        time = self.field_data.get_global_data()['time'][2]
        self.assertAlmostEqual(time, 2)

    def test_return_global_data_names(self):
        names = self.field_data.global_field_names
        self.assertIn("load", names)
        self.assertIn("displacement", names)

    def test_confirm_global_data(self):
        load_data = self.field_data.get_global_data()['load']
        disp_data = self.field_data.get_global_data()['displacement']
        goal_load = np.array([0, 10, 21])
        goal_disp = np.array([0, .01, .02])
        self.assertTrue(np.allclose(load_data, goal_load))
        self.assertTrue(np.allclose(disp_data, goal_disp))
