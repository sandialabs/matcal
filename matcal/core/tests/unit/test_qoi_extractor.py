from matcal.core.state import State
from matcal.core.utilities import _time_interpolate
import numpy as np

from matcal.core.qoi_extractor import (DataSpecificExtractorWrapper, MaxExtractor, 
                                       QoIExtractorBase, InterpolatingExtractor,
                                       ReturnPassedDataExtractor, StateSpecificExtractorWrapper, 
                                       UserDefinedExtractor)
from matcal.core.tests.MatcalUnitTest import MatcalUnitTest
from matcal.core.data import convert_dictionary_to_data


class TestMaxExtractor(MatcalUnitTest):

    def setUp(self):
        super().setUp(__file__)

        data = {"x": np.append(np.linspace(0, 1, 21), np.linspace(0.95, 0, 20)),
                "y": np.linspace(0, 40, 41),
                "z": np.linspace(0, 20, 41)}

        self.test_data_one_peak = convert_dictionary_to_data(data)

        data_2 = {"x": np.append(np.append(np.linspace(0, 1, 21), np.linspace(0.95, 0, 20)),
                                 np.linspace(0, 1, 20)),
                  "y": np.append(np.linspace(0, 40, 41), np.linspace(39, 10, 20)),
                  "z": np.append(np.linspace(0, 20, 41), np.linspace(19.5, 0, 20))}

        self.test_data_two_peak = convert_dictionary_to_data(data_2)

    def test_initialize(self):
        extractor = MaxExtractor("x")
        extractor = MaxExtractor("x", 0)
        extractor = MaxExtractor("x", max_index=-1)

    def test_raise_invalid_field_types(self):
        with self.assertRaises(TypeError):
            extractor = MaxExtractor(None)
            extractor = MaxExtractor(1)

    def test_raise_invalid_index_passed(self):
        with self.assertRaises(TypeError):
            extractor = MaxExtractor('x', None)
            extractor = MaxExtractor('x', 'x')

    def test_extract_max_single_peak(self):
        extractor = MaxExtractor("x")
        extracted_data_1 = extractor.calculate(self.test_data_one_peak, self.test_data_one_peak,
                                               self.test_data_one_peak.field_names)

        self.assertAlmostEqual(extracted_data_1["x"], 1)
        self.assertAlmostEqual(extracted_data_1["y"], 20)
        self.assertAlmostEqual(extracted_data_1["z"], 10)

    def test_extract_max_multi_peak_first_max(self):
        extractor = MaxExtractor("x")
        extracted_data_1 = extractor.calculate(self.test_data_two_peak, self.test_data_two_peak,
                                               self.test_data_two_peak.field_names)

        self.assertAlmostEqual(extracted_data_1["x"], 1)
        self.assertAlmostEqual(extracted_data_1["y"], 20)
        self.assertAlmostEqual(extracted_data_1["z"], 10)

    def test_extract_max_multi_peak_last_max(self):
        extractor = MaxExtractor("x", max_index=-1)
        extracted_data = extractor.calculate(self.test_data_two_peak, self.test_data_two_peak,
                                             self.test_data_two_peak.field_names)

        self.assertAlmostEqual(extracted_data["x"][0], 1)
        self.assertAlmostEqual(extracted_data["y"][0], 10)
        self.assertAlmostEqual(extracted_data["z"][0], 0)

    def test_extract_max_data_of_len_one(self):
        extractor = MaxExtractor("x")
        data_len_one_dict = {"x":np.ones(1)}
        data_len_one = convert_dictionary_to_data(data_len_one_dict)

        extracted_data = extractor.calculate(data_len_one, data_len_one,
                                             data_len_one.field_names)

        self.assertAlmostEqual(extracted_data["x"][0], 1)


class TestUserDefinedExtractor(MatcalUnitTest):

    def setUp(self):
        super().setUp(__file__)

        data = {"x": np.append(np.linspace(0, 1, 21), np.linspace(0.95, 0, 20)),
                "y": np.linspace(0, 40, 41),
                "z": np.linspace(0, 20, 41)}

        self.test_data_one_peak = convert_dictionary_to_data(data)

        data_2 = {"x": np.append(np.append(np.linspace(0, 1, 21), np.linspace(0.95, 0, 20)),
                                 np.linspace(0, 1, 20)),
                  "y": np.append(np.linspace(0, 40, 41), np.linspace(39, 10, 20)),
                  "z": np.append(np.linspace(0, 20, 41), np.linspace(19.5, 0, 20))}

        self.test_data_two_peak = convert_dictionary_to_data(data_2)

    def test_initialize(self):
        def func(x, y, fields):
            return x
        with self.assertRaises(TypeError):
            extractor = UserDefinedExtractor(func, None)

    def test_raise_invalid_field_types(self):
        with self.assertRaises(TypeError):
            extractor = UserDefinedExtractor(None)
            extractor = UserDefinedExtractor(1)

    def test_extract_max_single_peak_dict(self):
        def func(x, y, fields):
            extracted_data = x[x["x"] == x["x"].max()]
            extracted_data_dict = {}
            for field in extracted_data.field_names:
                extracted_data_dict[field] = extracted_data[field]

            return extracted_data_dict

        extractor = UserDefinedExtractor(func,'x')

        extracted_data_1 = extractor.calculate(self.test_data_one_peak, self.test_data_one_peak,
                                               self.test_data_one_peak.field_names)
        self.assertAlmostEqual(extracted_data_1["x"], 1)
        self.assertAlmostEqual(extracted_data_1["y"], 20)
        self.assertAlmostEqual(extracted_data_1["z"], 10)


    def test_raise_error_when_funciton_errors(self):
        def func(x, y, fields):
            raise RuntimeError("I'm a bad function, I only cause errors")

        extractor = UserDefinedExtractor(func,'x')

        with self.assertRaises(RuntimeError):
            extracted_data_1 = extractor.calculate(self.test_data_one_peak, 
                                                self.test_data_one_peak,
                                               self.test_data_one_peak.field_names)



    def test_extract_max_single_peak_dataframe(self):
        def func(x, y, fields):
            extracted_data = x[x["x"] == x["x"].max()]
            return extracted_data

        extractor = UserDefinedExtractor(func, 'x')

        extracted_data_1 = extractor.calculate(self.test_data_one_peak, self.test_data_one_peak,
                                               self.test_data_one_peak.field_names)

        self.assertAlmostEqual(extracted_data_1["x"], 1)
        self.assertAlmostEqual(extracted_data_1["y"], 20)
        self.assertAlmostEqual(extracted_data_1["z"], 10)

    def test_extract_max_single_peak_numpy_return(self):
        def func(x, y, fields):
            extracted_data = x[x["x"] == x["x"].max()]

            return np.array(extracted_data.tolist())

        extractor = UserDefinedExtractor(func, 'x')
        with self.assertRaises(TypeError):
            extracted_data_1 = extractor.calculate(self.test_data_one_peak, self.test_data_one_peak,
                                                   self.test_data_one_peak.field_names)

    def test_required_fields(self):
        def func(x, y, fields):
            extracted_data = x[x["x"] == x["x"].max()]

            return np.array(extracted_data.tolist())

        extractor = UserDefinedExtractor(func, 'x', 'y', 't')
        self.assertIn('x', extractor.required_experimental_data_fields)
        self.assertIn('y', extractor.required_experimental_data_fields)
        self.assertIn('t', extractor.required_experimental_data_fields)


class TestInterpolatingExtractor(MatcalUnitTest):

    def setUp(self):
        super().setUp(__file__)

        ref_data = {"x": np.linspace(0, 1, 21),
                    "y": np.linspace(0, 40, 21),
                    "z": np.linspace(0, 20, 21)}

        eval_data = {"x": np.linspace(0, 1, 11),
                     "y": np.linspace(0, 20, 11),
                     "z": np.linspace(0, 10, 11)}

        self.answer_data = {"x": np.linspace(0, 1, 21),
                            "y": np.linspace(0, 20, 21),
                            "z": np.linspace(0, 10, 21)}

        self.ref_data = convert_dictionary_to_data(ref_data)
        self.eval_data = convert_dictionary_to_data(eval_data)

    def test_initialize(self):
        extractor = InterpolatingExtractor("x")
        extractor = InterpolatingExtractor("x")

    def test_raise_invalid_field_types(self):
        with self.assertRaises(TypeError):
            extractor = InterpolatingExtractor(None)
            extractor = InterpolatingExtractor(1)

    def test_extract_interpolated_data(self):
        interpolator = InterpolatingExtractor("x")
        interped_eval_data = interpolator.calculate(self.eval_data, self.ref_data, self.ref_data.field_names)
        for field in interped_eval_data.field_names:
            comparison_field_data = self.answer_data[field]
            self.assertTrue(np.allclose(comparison_field_data, interped_eval_data[field]))

    def test_extract_exterpolated_data(self):
        interpolator = InterpolatingExtractor("x", right=10)
        extrap_eval_data = {"x": np.linspace(0, 0.5, 11),
                           "y": np.linspace(0, 10, 11),
                           "z": np.linspace(0, 5, 11)}
        extrap_eval_data = convert_dictionary_to_data(extrap_eval_data)
        extrap_answer = {"x": np.linspace(0, 1, 21),
                         "y": np.concatenate((np.linspace(0, 10, 11), 10*np.ones(10))),
                         "z": np.concatenate((np.linspace(0, 5, 11), 10*np.ones(10)))}
        interped_eval_data = interpolator.calculate(extrap_eval_data, self.ref_data, self.ref_data.field_names)
        for field in interped_eval_data.field_names:
            comparison_field_data = extrap_answer[field]
            self.assertTrue(np.allclose(comparison_field_data, interped_eval_data[field]))


class TestReturnPassedDataExtractor(MatcalUnitTest):

    def setUp(self):
        super().setUp(__file__)

        ref_data = {"x": np.linspace(0, 1, 21),
                    "y": np.linspace(0, 40, 21),
                    "z": np.linspace(0, 20, 21)}

        eval_data = {"x": np.linspace(0, 1, 21),
                     "y": np.linspace(0, 20, 21),
                     "z": np.linspace(0, 10, 21)}

        self.ref_data = convert_dictionary_to_data(ref_data)
        self.eval_data = convert_dictionary_to_data(eval_data)

    def test_initialize(self):
        extractor = ReturnPassedDataExtractor()

    def test_raise_invalid_field_types(self):
        with self.assertRaises(TypeError):
            extractor = ReturnPassedDataExtractor(None)
            extractor = ReturnPassedDataExtractor(1)
            extractor = ReturnPassedDataExtractor("x")

    def test_return_passed_data_extractor(self):
        extractor = ReturnPassedDataExtractor()
        eval_data = extractor.calculate(self.eval_data, self.ref_data, self.ref_data.field_names)
        for field in eval_data.field_names:
            self.assertTrue(np.allclose(self.eval_data[field], eval_data[field]))



class TestStateSpecificExtractorWrapper(MatcalUnitTest):

    def setUp(self):
        super().setUp(__file__)

    def test_add_and_use_extractor_one_state(self):
        ssew = StateSpecificExtractorWrapper()
        state_name = 'state1'
        state = State(state_name, a=1)
        qoi_name = 'temp'
        ext = MaxExtractor(qoi_name)
        goal_max = 13
        raw_data = {'time': [0,1,2,3], 'temp': [10, 11, 12, goal_max]}
        data = convert_dictionary_to_data(raw_data)
        data.set_state(state)
        ssew.add(data, ext)
        qoi = ssew.calculate(data, data, 'temp')
        self.assertAlmostEqual(qoi[qoi_name], goal_max)

    def test_get_required_fields(self):
        ssew = StateSpecificExtractorWrapper()
        state_name = 'state1'
        state = State(state_name, a=1)
        qoi_name = 'temp'
        ext = MaxExtractor(qoi_name)
        goal_max = 13
        raw_data = {'time': [0,1,2,3], 'temp': [10, 11, 12, goal_max]}
        data = convert_dictionary_to_data(raw_data)
        data.set_state(state)
        ssew.add(data, ext)
        req_fields = ssew.required_experimental_data_fields
        self.assertEqual(len(req_fields), 1)
        self.assertEqual(req_fields[0], qoi_name)

    def test_add_and_use_two_states(self):
        ssew = StateSpecificExtractorWrapper()
        state1 = State('s1', a=1)
        state2 = State('s2', a=2)
        states = [state1, state2]
        n_points = 10
        qoi_names = ['time', 'temp', 'disp']
        datas = [{}, {}]
        for data_idx, data in enumerate(datas):
            for name_idx, name in enumerate(qoi_names):
                data[name] = (np.linspace(0, 10, n_points) + name_idx) * (data_idx+1)
            data = convert_dictionary_to_data(data)
            data.set_state(states[data_idx])
            datas[data_idx] = data

        ex1 = MaxExtractor(qoi_names[1])
        ex2 = MaxExtractor(qoi_names[2])
        ssew.add(datas[0], ex1)
        ssew.add(datas[1], ex2)
        qoi1 = ssew.calculate(datas[0], datas[0], None)
        qoi2 = ssew.calculate(datas[1], datas[1], None)
        self.assertAlmostEqual(qoi1[qoi_names[1]], 11)
        self.assertAlmostEqual(qoi2[qoi_names[2]], 24)


class TestDataSpecificExtractorWrapper(MatcalUnitTest):

    def setUp(self):
        super().setUp(__file__)

    def _make_datas(self, num_datas, qoi_names, num_points=10):
        datas = []
        for data_idx in range(num_datas):
            current_data = {}
            for name_idx, name in enumerate(qoi_names):
                current_data[name] = (data_idx + 1) * (np.linspace(0, 10, num_points) + name_idx)
            datas.append(convert_dictionary_to_data(current_data))
        return datas
    
    def _make_states(self, num_states):
        states = []
        for i in range(num_states):
            states.append(State(f'state_{i}'))
        return states

    def test_add_and_use_once(self):
        dsew = DataSpecificExtractorWrapper()
        datas = self._make_datas(1, ['time', 'T'])
        ex = MaxExtractor("T")
        dsew.add(datas[0], ex)
        qoi  = dsew.calculate(datas[0], datas[0], None)
        val = qoi['T']
        self.assertAlmostEqual(val, 11)

    def test_add_and_use_multiple_data_one_state(self):
        dsew = DataSpecificExtractorWrapper()
        qoi_names = ['time']
        n_datas = 4
        exs = []
        for i in range(n_datas):
            cur_qoi = f"QOI-{i+1}"
            qoi_names.append(cur_qoi)
            exs.append(MaxExtractor(cur_qoi))
        datas = self._make_datas(n_datas, qoi_names)
        for d, e in zip(datas, exs):
            dsew.add(d, e)
        for i, d in enumerate(datas):
            cur_qoi = f"QOI-{i+1}"
            qoi = dsew.calculate(d,d, None)
            val = qoi[cur_qoi]
            goal_val = float((i+1) * (10 + i + 1))
            self.assertAlmostEqual(val, goal_val)

    def test_add_and_use_multiple_data_unique_state(self):
        dsew = DataSpecificExtractorWrapper()
        qoi_names = ['time']
        n_datas = 4
        states = self._make_states(n_datas)
        exs = []
        for i in range(n_datas):
            cur_qoi = f"QOI-{i+1}"
            qoi_names.append(cur_qoi)
            exs.append(MaxExtractor(cur_qoi))
        datas = self._make_datas(n_datas, qoi_names)
        for state, data in zip(states, datas):
            data.set_state(state)
            
        for d, e in zip(datas, exs):
            dsew.add(d, e)
        for i, d in enumerate(datas):
            cur_qoi = f"QOI-{i+1}"
            qoi = dsew.calculate(d,d, None)
            val = qoi[cur_qoi]
            goal_val = float((i+1) * (10 + i + 1))
            self.assertAlmostEqual(val, goal_val)

    def test_add_and_use_multiple_data_two_state(self):
        dsew = DataSpecificExtractorWrapper()
        qoi_names = ['time']
        n_datas = 4
        states = self._make_states(2)
        exs = []
        for i in range(n_datas):
            cur_qoi = f"QOI-{i+1}"
            qoi_names.append(cur_qoi)
            exs.append(MaxExtractor(cur_qoi))
        datas = self._make_datas(n_datas, qoi_names)
        for data_i, data in enumerate(datas):
            data.set_state(states[data_i%2])
            
        for d, e in zip(datas, exs):
            dsew.add(d, e)
        for i, d in enumerate(datas):
            cur_qoi = f"QOI-{i+1}"
            qoi = dsew.calculate(d,d, None)
            val = qoi[cur_qoi]
            goal_val = float((i+1) * (10 + i + 1))
            self.assertAlmostEqual(val, goal_val)


class TestTimeInterpolate(MatcalUnitTest):

    def setUp(self):
        super().setUp(__file__)

    def test_time_interpolate_1d_array(self):
        x = np.linspace(0,10, 1000)
        x_interp = np.linspace(0,10,103)

        y = np.sin(x)
        y_goal = np.sin(x_interp)

        res = _time_interpolate(x_interp, x, y)

        self.assert_close_arrays(res, y_goal)
    
    def test_time_interpolate_1d_data_with_repeats(self):
        x = np.linspace(0,10, 1000)
        x_w_extras = np.append(x, np.array([0,10]))
        x_interp = np.linspace(0,10,103)
        
        y = np.sin(x_w_extras)
        y_goal = np.sin(x_interp)
        
        res = _time_interpolate(x_interp, x_w_extras, y)
        self.assert_close_arrays(res, y_goal)

    def test_when_single_data_point_is_passed_return_the_point(self):
        x = np.array([2])
        y = np.array([50])
        
        x_interp = np.array([1000])
        
        res = _time_interpolate(x_interp, x, y)
        self.assert_close_arrays(res, y)
        
    def test_use_linear_interpolation_when_only_two_points_exist(self):
        x = np.array([0, 1])
        y = np.array([10, 20])
        
        x_interp = np.linspace(0, 1, 20)
        y_goal = 10 + 10 * x_interp
        
        res = _time_interpolate(x_interp, x, y)
        self.assert_close_arrays(res, y_goal)
        
    def test_use_quad_interpolation_when_greater_than_two_points_exist(self):
        x = np.array([0, 1, 2])
        y = np.array([0, 1, 4])
        
        x_interp = np.linspace(0, 1, 20)
        y_goal = np.power(x_interp, 2)
        
        res = _time_interpolate(x_interp, x, y)
        self.assert_close_arrays(res, y_goal)