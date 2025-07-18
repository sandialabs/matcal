import numpy as np
import os

from matcal.core.data_importer import FileData
from matcal.core.tests.MatcalUnitTest import MatcalUnitTest

from matcal.exodus.data_importer import ExodusFieldDataSeriesImporter
from matcal.exodus.tests.utilities import test_support_files_dir


class TestExodusFieldDataSeriesImporter(MatcalUnitTest):
    def setUp(self):
        super().setUp(__file__)
        self.test_support_files_path = os.path.join(test_support_files_dir)
        filename = os.path.join(self.test_support_files_path, "simple_cubes.e")
        self.efi = ExodusFieldDataSeriesImporter(filename)

    def test_raise_error_no_file(self):
        with self.assertRaises(ExodusFieldDataSeriesImporter.FieldDataDataSeriesMissingPathObject):
            ExodusFieldDataSeriesImporter("not_a_file")

    def test_get_number_of_frames(self):
        self.assertEqual(self.efi.number_of_frames, 21)

    def test_confirm_global_variable_names(self):
        keys = ["time", "kinetic_energy", "internal_energy", "external_energy",
                "momentum_x", "momentum_y", "momentum_z", "timestep"]
        for key in keys:
            self.assertIn(key, self.efi.global_field_names)

    def test_confirm_number_of_blocks(self):
        self.assertEqual(self.efi.number_of_blocks, 2)

    def test_confirm_global_variable_values(self):
        goal_times = np.linspace(0, 1, 21)
        goal_dt = np.ones(21) * .05
        goal_dt[0] = 0.
        goal_internal_energy = np.array([0.0, 6.00964e-05,
                                         0.000240387,
                                         0.000540878,
                                         0.000961577,
                                         0.00150249,
                                         0.00216365,
                                         0.00294505,
                                         0.00384673,
                                         0.00486871,
                                         0.00601101,
                                         0.00727367,
                                         0.00865672,
                                         0.0101602,
                                         0.0117841,
                                         0.0135286,
                                         0.0153936,
                                         0.0173792,
                                         0.0194855,
                                         0.0217126,
                                         0.0240603])
        s_global_data = self.efi.get_global_data();
        self.assert_close_arrays(s_global_data['time'], goal_times)
        self.assert_close_arrays(s_global_data['timestep'], goal_dt)
        self.assert_close_arrays(s_global_data['internal_energy'], goal_internal_energy)

    def test_get_frame_information_time_0(self):
        fields_of_interest = ["stress_xx", "stress_xy", "stress_yy", "stress_yz", "stress_zx", "stress_zz"]
        frame_element_data = self.efi.get_frame(0)
        goal_dict = self._make_goal_dict(0, fields_of_interest)
        for field in fields_of_interest:
            test_data = frame_element_data[field]
            goal_data = goal_dict[field]
            self.assert_close_arrays(test_data, goal_data)

    def test_get_frame_information_time_10(self):
        fields_of_interest = ["stress_xx", "stress_xy", "stress_yy", "stress_yz", "stress_zx", "stress_zz"]
        frame_element_data = self.efi.get_frame(10)
        goal_dict = self._make_goal_dict(10, fields_of_interest)
        for field in fields_of_interest:
            frame = frame_element_data[field]
            goal = goal_dict[field]

            self.assert_close_arrays(frame, goal)

    def test_get_frame_information_time_20(self):
        fields_of_interest = ["stress_xx", "stress_xy", "stress_yy", "stress_yz", "stress_zx", "stress_zz"]
        frame_element_data = self.efi.get_frame(20)
        goal_dict = self._make_goal_dict(20, fields_of_interest)
        for field in fields_of_interest:
            self.assert_close_arrays(frame_element_data[field], goal_dict[field])

    def test_get_field_value_for_block_for_all_time(self):
        block_index = 1
        block_1_stress_yy = self.efi.get_values_for_all_time(block_index, 'stress_yy')
        goal = np.array([0.00000000e+00, 2.41062902e-05, 4.53431335e-05,
                         5.90541582e-05, 8.01294836e-05, 1.04504909e-04, 1.33823535e-04,
                         1.70007708e-04, 2.10679101e-04, 2.57641047e-04, 3.10172952e-04,
                         3.68136539e-04, 4.32040780e-04, 5.01376674e-04, 5.76419162e-04,
                         6.57103287e-04, 7.43331928e-04, 8.35216694e-04, 9.32654183e-04,
                         1.03566792e-03, 1.14425083e-03])
        self.assert_close_arrays(block_1_stress_yy, goal)

    def test_raise_error_bad_frame_index(self):
        with self.assertRaises(ExodusFieldDataSeriesImporter.FieldDataDataSeriesBadFrameIndex):
            self.efi.get_frame(-2)
            self.efi.get_frame(21)

    def _make_goal_dict(self, index, fields_of_interest):
        goal_filename = os.path.join(self.test_support_files_path, f"element_data_{index}.csv")

        df = FileData(goal_filename)
        goal_dict = {}
        for field in fields_of_interest:
            goal_dict[field] = self._get_information_array(df, field)
        return goal_dict

    def _get_information_array(self, data_frame, field_name):
        return [data_frame[field_name]]

    def test_get_field_value_for_multi_element_block(self):
        exodus_filename = os.path.join(self.test_support_files_path,"vfm_single_block_multiple_elements.e")

        efi = ExodusFieldDataSeriesImporter(exodus_filename)
        stress_yy = efi.get_all_element_values_for_all_time(1, 'first_pk_stress_yy')
        goal = []
        element_entry = [0., 0.00449966, 0.0112489, 0.0213722, 0.03655585, 0.05932846,
                         0.09348091, 0.14469504, 0.22148364, 0.3365934, 0.50909442, 0.76748071,
                         1.15424812, 1.60095784, 2.04766757, 2.4943773, 2.94108702, 3.38779675,
                         3.83450648, 4.2812162, 4.47181451]

        for i in range(12):
            goal.append(element_entry)
        goal = np.array(goal)
        self.assert_close_arrays(goal, stress_yy)


class TestExodusFieldDataSeriesImporterWithDecomp(MatcalUnitTest):

    def setUp(self):
        super().setUp(__file__)

    def test_decomposed_mesh_files(self):
        composed_mesh_name = os.path.join(test_support_files_dir, "cube8ele_decomp.g")
        efi = ExodusFieldDataSeriesImporter(composed_mesh_name)
        self.assertEqual(8, efi.number_of_elements)
        os.remove(composed_mesh_name)

    def test_get_field_value_for_multi_element_block_parallel(self):
        exodus_filename = os.path.join(test_support_files_dir, 
                                       "vfm_single_block_multiple_elements.e")
        efi = ExodusFieldDataSeriesImporter(exodus_filename, n_cores=5)
        stress_yy = efi.get_all_element_values_for_all_time(1, 'first_pk_stress_yy')
        goal = []
        element_entry = [0., 0.00449966, 0.0112489, 0.0213722, 0.03655585, 0.05932846,
                         0.09348091, 0.14469504, 0.22148364, 0.3365934, 0.50909442, 0.76748071,
                         1.15424812, 1.60095784, 2.04766757, 2.4943773, 2.94108702, 3.38779675,
                         3.83450648, 4.2812162, 4.47181451]

        for i in range(12):
            goal.append(element_entry)
        goal = np.array(goal)
        self.assert_close_arrays(goal, stress_yy)