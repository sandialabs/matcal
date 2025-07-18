from collections import OrderedDict
import numpy as np
import os

from matcal.core.boundary_condition_calculators import (DISPLACEMENT_KEY, ENG_STRAIN_KEY, 
    TEMPERATURE_KEY, TIME_KEY, BoundaryConditionDeterminationError, max_state_values, 
    ROTATION_KEY, ROTATION_RATE_KEY, DISPLACEMENT_RATE_KEY, STRAIN_RATE_KEY, 
    TRUE_STRAIN_KEY, 
    get_displacement_function_from_load_displacement_data_collection, 
    get_displacement_function_from_strain_data_collection, 
    get_temperature_function_from_data_collection, 
    get_rotation_function_from_data_collection)
from matcal.core.data import DataCollection, convert_dictionary_to_data
from matcal.core.data_importer import FileData
from matcal.core.state import State
from matcal.core.tests.MatcalUnitTest import MatcalUnitTest


def make_single_state_displacement_data_collection(**state_variables):
    state_variables[DISPLACEMENT_RATE_KEY] = 1e-3
    state = State("state", **state_variables)

    data_dict1  = {DISPLACEMENT_KEY:np.linspace(0,1, 10)}
    data1 = convert_dictionary_to_data(data_dict1)
    data1.set_state(state)
    data1.set_name("data 1")

    data_dict2  = {DISPLACEMENT_KEY:np.linspace(-1,1, 10)}
    data2 = convert_dictionary_to_data(data_dict2)
    data2.set_state(state)
    data2.set_name("data 2")

    data_dict3  = {DISPLACEMENT_KEY:np.linspace(-1,2, 10)}
    data3 = convert_dictionary_to_data(data_dict3)
    data3.set_state(state)
    data3.set_name("data 3")
    
    data_dict4  = {DISPLACEMENT_KEY:np.linspace(-1,1.5, 10)}
    data4 = convert_dictionary_to_data(data_dict4)
    data4.set_state(state)
    data4.set_name("data 4")
    
    dc = DataCollection('test', data1, data2, data3, data4)

    return dc, state


def make_single_state_time_strain_data_collection(**state_variables):
    state_variables[STRAIN_RATE_KEY] = 1e-3
    state = State("state", **state_variables)

    data_dict1  = {ENG_STRAIN_KEY:np.linspace(0,1, 10), TIME_KEY:np.linspace(0,100,10)}
    data1 = convert_dictionary_to_data(data_dict1)
    data1.set_state(state)
    data1.set_name("data 1")

    data_dict2  = {ENG_STRAIN_KEY:np.linspace(-1,1, 10), TIME_KEY:np.linspace(0,101,10)}
    data2 = convert_dictionary_to_data(data_dict2)
    data2.set_state(state)
    data2.set_name("data 2")

    data_dict3  = {ENG_STRAIN_KEY:np.linspace(-1,2, 10), TIME_KEY:np.linspace(0,102,10)}
    data3 = convert_dictionary_to_data(data_dict3)
    data3.set_state(state)
    data3.set_name("data 3")
    
    data_dict4  = {ENG_STRAIN_KEY:np.linspace(-1,1.5, 10), TIME_KEY:np.linspace(0,99,10)}
    data4 = convert_dictionary_to_data(data_dict4)
    data4.set_state(state)
    data4.set_name("data 4")
    
    dc = DataCollection('test', data1, data2, data3, data4)

    return dc, state


def engineering_strain_to_true_strain(engineering_strain):
    return np.log(engineering_strain+1)


def make_single_state_time_true_strain_data_collection(**state_variables):
    dc, state = make_single_state_time_strain_data_collection(**state_variables)
    for state_iter in dc:
        for index, data in enumerate(dc[state_iter]):
            data.rename_field("engineering_strain", "true_strain")
            data["true_strain"] = engineering_strain_to_true_strain(data["true_strain"])
            dc[state_iter][index] = data

    return dc, state 


def make_dual_state_time_true_strain_data_collection(**state_variables):
    dc, state1 = make_single_state_time_strain_data_collection(**state_variables)
    for state_iter in dc:
        for index, data in enumerate(dc[state_iter]):
            data.rename_field("engineering_strain", "true_strain")
            data["true_strain"] = engineering_strain_to_true_strain(data["true_strain"])
            dc[state_iter][index] = data

    state_variables[STRAIN_RATE_KEY] = 1e2
    state2 = State("state", **state_variables)

    data_dict1  = {TRUE_STRAIN_KEY:np.linspace(0,1, 10), TIME_KEY:np.linspace(0,100,10)}
    data1 = convert_dictionary_to_data(data_dict1)
    data1.set_state(state2)
    data1.set_name("data 1")

    data_dict2  = {TRUE_STRAIN_KEY:np.linspace(-1,1, 10), TIME_KEY:np.linspace(0,101,10)}
    data2 = convert_dictionary_to_data(data_dict2)
    data2.set_state(state2)
    data2.set_name("data 2")

    data_dict3  = {TRUE_STRAIN_KEY:np.linspace(-1,2, 10), TIME_KEY:np.linspace(0,102,10)}
    data3 = convert_dictionary_to_data(data_dict3)
    data3.set_state(state2)
    data3.set_name("data 3")
    
    data_dict4  = {TRUE_STRAIN_KEY:np.linspace(-1,1.5, 10), TIME_KEY:np.linspace(0,99,10)}
    data4 = convert_dictionary_to_data(data_dict4)
    data4.set_state(state2)
    data4.set_name("data 4")

    return dc, state1, state2 

class TestDisplacementBoundaryConditionCalculator(MatcalUnitTest):

    def setUp(self):
        super().setUp(__file__)

    def test_max_state_values(self):
        dc, state = make_single_state_displacement_data_collection()
        data3 = dc[state][2]

        max_data_index, max_data, max_value_index = max_state_values(dc[state], DISPLACEMENT_KEY)

        self.assert_close_arrays(data3, max_data)
        self.assert_close_arrays(data3, dc[state][max_data_index])
        self.assertEqual(os.path.split(max_data.name)[-1], "data 3")
        self.assertEqual(max_value_index, np.argmax(data3[DISPLACEMENT_KEY]))

    def test_extract_displacement_function_from_time_strain_data(self):
        dc, state = make_single_state_time_strain_data_collection()
        data3 = dc[state][2]
        extensometer_length = 1.25

        disp_func = get_displacement_function_from_strain_data_collection(dc, state, 
                                                                          scale_factor=extensometer_length)
        
        self.assertAlmostEqual(disp_func[DISPLACEMENT_KEY][-1], 
                               data3[ENG_STRAIN_KEY][-1]*extensometer_length)
        self.assertAlmostEqual(disp_func[TIME_KEY][-1], data3[TIME_KEY][-1])

    def test_extract_displacement_function_from_eng_strain_and_strain_rate_data(self):
        dc, state = make_single_state_time_strain_data_collection()
        data3 = dc[state][2]
        dc.remove_field("time")
        extensometer_length = 1.25

        disp_func = get_displacement_function_from_strain_data_collection(dc, state, scale_factor=extensometer_length)
        self.assertAlmostEqual(disp_func[DISPLACEMENT_KEY][-1],
                               data3[ENG_STRAIN_KEY][-1]*extensometer_length)
        self.assertAlmostEqual(disp_func[TIME_KEY][-1], data3[ENG_STRAIN_KEY][-1]/state[STRAIN_RATE_KEY])
 
    def test_extract_displacement_function_from_displacement_and_displacement_rate_data(self):
        dc, state = make_single_state_displacement_data_collection()
        data3 = dc[state][2]
        extensometer_length=1.25
        disp_func = get_displacement_function_from_load_displacement_data_collection(dc, state, scale_factor=extensometer_length)

        self.assertAlmostEqual(disp_func[DISPLACEMENT_KEY][-1], data3[DISPLACEMENT_KEY][-1]*extensometer_length)
        self.assertAlmostEqual(disp_func[TIME_KEY][-1], data3[DISPLACEMENT_KEY][-1]/state[DISPLACEMENT_RATE_KEY])

    def test_displacement_function_from_eng_strain_and_strain_rate_data_same_as_function_from_disp_disp_rate_data(self):
        dc_disp, state_disp = make_single_state_displacement_data_collection()
        dc_strain, state_strain = make_single_state_time_strain_data_collection()
        dc_strain.remove_field("time")

        extensometer_length = 1.25
        disp_func_from_disp = get_displacement_function_from_load_displacement_data_collection(dc_disp, state_disp, scale_factor=extensometer_length)
        disp_func_from_strain = get_displacement_function_from_strain_data_collection(dc_strain, state_strain, scale_factor=extensometer_length)

        self.assert_close_arrays(disp_func_from_strain, disp_func_from_disp)

    def test_displacement_function_from_strain_function(self):
        dc, state = make_single_state_time_strain_data_collection()
        state._state_variables = OrderedDict()
        data3 = dc[state][2]
        dc.remove_field("time")
        extensometer_length = 1.25

        disp_func = get_displacement_function_from_strain_data_collection(dc, state, 
                                                                          scale_factor=extensometer_length)
        self.assertAlmostEqual(disp_func[DISPLACEMENT_KEY][-1],
                               data3[ENG_STRAIN_KEY][-1]*extensometer_length)
        self.assertAlmostEqual(disp_func[TIME_KEY][-1], 1.0)

    def test_displacement_function_from_true_strain_function_with_rate(self):

        dc_true_strain, state_true_strain = make_single_state_time_true_strain_data_collection()
        data3_true = dc_true_strain[state_true_strain][2]
        dc_true_strain.remove_field("time")
        dc_eng_strain, state_eng_strain = make_single_state_time_strain_data_collection()
        dc_eng_strain.remove_field("time")

        true_strain_func = get_displacement_function_from_strain_data_collection(dc_true_strain, 
                                                                                 state_true_strain, scale_factor=1)
        engineering_strain_func = get_displacement_function_from_strain_data_collection(dc_eng_strain, 
                                                                                        state_eng_strain, 
                                                                                        scale_factor=1)

        
        self.assert_close_arrays(true_strain_func, engineering_strain_func)
        gold_function = np.array([[0, 0], [(np.exp(data3_true[TRUE_STRAIN_KEY][-1])-1)/state_true_strain[STRAIN_RATE_KEY], (np.exp(data3_true[TRUE_STRAIN_KEY][-1])-1)]])
        self.assert_close_arrays(true_strain_func, gold_function)

    def test_displacement_function_from_true_strain_function_rate_independent(self):
        dc_true_strain, state_true_strain = make_single_state_time_true_strain_data_collection()
        state_true_strain._state_variables = OrderedDict() 
        data3_true = dc_true_strain[state_true_strain][2]
        dc_true_strain.remove_field("time")
        dc_eng_strain, state_eng_strain = make_single_state_time_strain_data_collection()
        state_eng_strain._state_variables = OrderedDict()
        dc_eng_strain.remove_field("time")

        true_strain_func = get_displacement_function_from_strain_data_collection(dc_true_strain, 
                                                                                 state_true_strain, 
                                                                                 scale_factor=1)
        engineering_strain_func = get_displacement_function_from_strain_data_collection(dc_eng_strain, 
                                                                                        state_eng_strain, 
                                                                                        scale_factor=1)

        gold_function_rate_independent = np.array([[0, 0], [1.0, (np.exp(data3_true[TRUE_STRAIN_KEY][-1])-1)]])
        self.assert_close_arrays(true_strain_func, gold_function_rate_independent)
        self.assert_close_arrays(engineering_strain_func, engineering_strain_func)

    def test_displacement_function_from_true_strain_function_rate_independent_multi_datasets_per_state_two_states(self):
        dc_true_strain, state1_true_strain, state2_true_strain = make_dual_state_time_true_strain_data_collection()
        dc_true_strain.remove_field("time")
        state1_gold_data = dc_true_strain[state1_true_strain.name][2]
        state2_gold_data = dc_true_strain[state2_true_strain.name][2]
        state1_true_strain._state_variables = OrderedDict()
        state2_true_strain._state_variables = OrderedDict()
        
        disp_func_state1= get_displacement_function_from_strain_data_collection(dc_true_strain, 
                                                                                state1_true_strain, 
                                                                                scale_factor=1)
        gold_function_state1 = np.array([[0, 0], [1.0, (np.exp(state1_gold_data[TRUE_STRAIN_KEY][-1])-1)]])
        self.assert_close_arrays(disp_func_state1, gold_function_state1)

        disp_func_state2 = get_displacement_function_from_strain_data_collection(dc_true_strain, 
                                                                                 state2_true_strain, 
                                                                                 scale_factor=1)
        gold_function_state2 = np.array([[0, 0], [1.0, (np.exp(state2_gold_data[TRUE_STRAIN_KEY][-1])-1)]])
        self.assert_close_arrays(disp_func_state2, gold_function_state2)

    def test_verify_data_collection_not_modified(self):
        dc_true_strain, state1_true_strain, state2_true_strain = make_dual_state_time_true_strain_data_collection()
        disp_func_state1= get_displacement_function_from_strain_data_collection(dc_true_strain, 
                                                                                state1_true_strain, 
                                                                                scale_factor=1)
        disp_func_state2 = get_displacement_function_from_strain_data_collection(dc_true_strain, 
                                                                                 state2_true_strain, 
                                                                                 scale_factor=1)
        dc_true_strain_gold, state1_true_strain, state2_true_strain = make_dual_state_time_true_strain_data_collection()

        self.assertEqual(dc_true_strain, dc_true_strain_gold)

    def test_displacement_function_from_true_strain_function_rate_dependent_multi_datasets_per_state_two_states(self):
        dc_true_strain, state1_true_strain, state2_true_strain = make_dual_state_time_true_strain_data_collection()
        dc_true_strain.remove_field("time")
        state1_gold_data = dc_true_strain[state1_true_strain.name][2]
        state2_gold_data = dc_true_strain[state2_true_strain.name][2]
        
        disp_func_state1= get_displacement_function_from_strain_data_collection(dc_true_strain, 
                                                                                state1_true_strain, 
                                                                                scale_factor=1)
        max_disp_state1 = (np.exp(state1_gold_data[TRUE_STRAIN_KEY][-1])-1)
        gold_function_state1 = np.array([[0, 0], [max_disp_state1/state1_true_strain[STRAIN_RATE_KEY], max_disp_state1]])
        self.assert_close_arrays(disp_func_state1, gold_function_state1)

        disp_func_state2 = get_displacement_function_from_strain_data_collection(dc_true_strain, 
                                                                                 state2_true_strain, 
                                                                                 scale_factor=1)
        max_disp_state2 = (np.exp(state2_gold_data[TRUE_STRAIN_KEY][-1])-1)
        
        gold_function_state2 = np.array([[0, 0], [max_disp_state2/state2_true_strain[STRAIN_RATE_KEY], max_disp_state2]])
        self.assert_close_arrays(disp_func_state2, gold_function_state2)

    def test_displacement_function_from_true_strain_function_multi_datasets_per_state_two_states_with_time(self):
        dc_true_strain, state1_true_strain, state2_true_strain = make_dual_state_time_true_strain_data_collection()
        state1_gold_data = dc_true_strain[state1_true_strain.name][2]
        state2_gold_data = dc_true_strain[state2_true_strain.name][2]
       
        disp_func_state1= get_displacement_function_from_strain_data_collection(dc_true_strain, 
                                                                                state1_true_strain, 
                                                                                scale_factor=1)

        disp_vec_state1 = (np.exp(state1_gold_data[TRUE_STRAIN_KEY])-1)
        time_vec_state1 = state1_gold_data[TIME_KEY]
        gold_function_state1 = np.array([time_vec_state1, disp_vec_state1]).T

        self.assert_close_arrays(disp_func_state1, gold_function_state1)

        disp_func_state2 = get_displacement_function_from_strain_data_collection(dc_true_strain, 
                                                                                 state2_true_strain, 
                                                                                 scale_factor=1)

        disp_vec_state2 = (np.exp(state2_gold_data[TRUE_STRAIN_KEY])-1)
        time_vec_state2 = state2_gold_data[TIME_KEY]        
        gold_function_state2 = np.array([time_vec_state2, disp_vec_state2]).T
        self.assert_close_arrays(disp_func_state2, gold_function_state2)
        

class TestRotationBCFunctionCalculator(MatcalUnitTest):
    def setUp(self):
        super().setUp(__file__)
        self._rot_data = convert_dictionary_to_data({"grip_rotation":np.linspace(0, 500, 100)})
        self._rot_data2 = convert_dictionary_to_data({"grip_rotation":np.linspace(0, 520, 100)})
        self._rot_data3 = convert_dictionary_to_data({"grip_rotation":np.linspace(0, 495, 100)})

        self._rot_time_data = convert_dictionary_to_data({"grip_rotation":np.linspace(0, 500, 100), 
                                                        "time":np.linspace(0,10, 100)})
        self._rot_time_data2 = convert_dictionary_to_data({"grip_rotation":np.linspace(0, 520, 100), 
                                                        "time":np.linspace(0,9.9, 100)})
        self._rot_time_data3 = convert_dictionary_to_data({"grip_rotation":np.linspace(0, 495, 100), 
                                                        "time":np.linspace(0,11, 100)})
        self._rot_time_data4 = convert_dictionary_to_data({"grip_rotation":np.linspace(0, 495, 100), 
                                                        "time":np.linspace(0,1100, 100)})
        slow_rot_rate_state = State("slow rot rate")
        self._rot_time_data4.set_state(slow_rot_rate_state)
        self._rot_time_data5 = convert_dictionary_to_data({"grip_rotation":np.linspace(0, 500, 100), 
                                                        "time":np.linspace(0,1000, 100)})
        self._rot_time_data5.set_state(slow_rot_rate_state)

    def test_rotation_function_wrong_state(self):
        dc = DataCollection("test", self._rot_time_data)
        bad_state = State("not valid state")
        with self.assertRaises(BoundaryConditionDeterminationError):
            func = get_rotation_function_from_data_collection(dc, bad_state)

    def test_rotation_function_one_state_w_time(self):
        dc = DataCollection("test", self._rot_time_data)
        func = get_rotation_function_from_data_collection(dc, self._rot_time_data.state)
        goal_func = np.array([self._rot_time_data[TIME_KEY], self._rot_time_data[ROTATION_KEY]]).T
        self.assert_close_arrays(func, goal_func)

    def test_rotation_function_one_state_w_and_without_time(self):
        dc = DataCollection("test", self._rot_data)
        func = get_rotation_function_from_data_collection(dc, self._rot_data.state)
        goal_func = np.array([[0,0], [1.0, self._rot_data[ROTATION_KEY][-1]]])
        self.assert_close_arrays(func, goal_func)
    
    def test_rotation_function_one_state_w_out_time_rate_var(self):
        rate_state = State("state with rate")
        rate_state.update_state_variable(ROTATION_RATE_KEY, 1e-3)
        self._rot_data.set_state(rate_state)
        dc = DataCollection("test", self._rot_data)
        func = get_rotation_function_from_data_collection(dc, self._rot_data.state)
        goal_func = np.array([[0,0], [self._rot_data[ROTATION_KEY][-1]/rate_state[ROTATION_RATE_KEY], self._rot_data[ROTATION_KEY][-1]]])
        self.assert_close_arrays(func, goal_func)


class TestTemperatureBCFunctionCalculator(MatcalUnitTest):

    def setUp(self):
        super().setUp(__file__)
        self._temp_data = convert_dictionary_to_data({"temperature":np.linspace(298, 500, 100)})
        self._temp_data2 = convert_dictionary_to_data({"temperature":np.linspace(298, 520, 100)})
        self._temp_data3 = convert_dictionary_to_data({"temperature":np.linspace(298, 495, 100)})

        self._temp_time_data = convert_dictionary_to_data({"temperature":np.linspace(298, 500, 100), 
                                                        "time":np.linspace(0,10, 100)})
        self._temp_time_data2 = convert_dictionary_to_data({"temperature":np.linspace(298, 520, 100), 
                                                        "time":np.linspace(0,9.9, 100)})
        self._temp_time_data3 = convert_dictionary_to_data({"temperature":np.linspace(298, 495, 100), 
                                                        "time":np.linspace(0,11, 100)})
        

        self._temp_time_data4 = convert_dictionary_to_data({"temperature":np.linspace(298, 495, 100), 
                                                        "time":np.linspace(0,1100, 100)})
        slow_heat_rate_state = State("slow heat rate")
        self._temp_time_data4.set_state(slow_heat_rate_state)
        self._temp_time_data5 = convert_dictionary_to_data({"temperature":np.linspace(298, 500, 100), 
                                                        "time":np.linspace(0,1000, 100)})
        self._temp_time_data5.set_state(slow_heat_rate_state)

    def test_temperature_function_wrong_state(self):
        dc = DataCollection("test", self._temp_time_data)
        bad_state = State("not valid state")
        with self.assertRaises(BoundaryConditionDeterminationError):
            temp_func = get_temperature_function_from_data_collection(dc, bad_state)

    def test_temperature_function_one_state_w_time(self):
        dc = DataCollection("test", self._temp_time_data)
        temp_func = get_temperature_function_from_data_collection(dc, self._temp_time_data.state)
        goal_temp_func = np.array([self._temp_time_data[TIME_KEY], self._temp_time_data[TEMPERATURE_KEY]]).T
        self.assert_close_arrays(temp_func,goal_temp_func)

    def test_temperature_function_one_state_w_and_without_time(self):
        dc = DataCollection("test", self._temp_time_data, self._temp_data)
        with self.assertRaises(BoundaryConditionDeterminationError):
            temp_func = get_temperature_function_from_data_collection(dc, self._temp_time_data.state)
    
    def test_temperature_function_one_state_w_out_time(self):
        dc = DataCollection("test", self._temp_data)
        with self.assertRaises(BoundaryConditionDeterminationError):
            temp_func = get_temperature_function_from_data_collection(dc, self._temp_data.state)

    def test_temperature_function_one_state_repeats_w_time(self):
        dc = DataCollection("test", self._temp_time_data, self._temp_time_data2, self._temp_time_data3)
        min_temps = [np.min(self._temp_time_data[TEMPERATURE_KEY]), 
                            np.min(self._temp_time_data2[TEMPERATURE_KEY]), 
                            np.min(self._temp_time_data3[TEMPERATURE_KEY])]
        max_temps = [np.max(self._temp_time_data[TEMPERATURE_KEY]),
                            np.max(self._temp_time_data2[TEMPERATURE_KEY]), 
                            np.max(self._temp_time_data3[TEMPERATURE_KEY])]

        max_index = np.argmax(max_temps)
        temp_func = get_temperature_function_from_data_collection(dc, self._temp_data.state)
        time_values = dc[self._temp_data.state][max_index][TIME_KEY]
        temp_values = dc[self._temp_data.state][max_index][TEMPERATURE_KEY]
        goal_temp_func = np.array([time_values, temp_values]).T
        self.assert_close_arrays(temp_func,goal_temp_func)

    def test_temperature_function_one_state_repeats_w_out_time(self):
        dc = DataCollection("test", self._temp_data, self._temp_data2, self._temp_data3)
        with self.assertRaises(BoundaryConditionDeterminationError):
            temp_func = get_temperature_function_from_data_collection(dc, self._temp_data.state)

    def test_temperature_function_two_state_repeats_w_time(self):
        dc = DataCollection("test", self._temp_time_data, self._temp_time_data2, self._temp_time_data3, 
                            self._temp_time_data4, self._temp_time_data5)
        min_temps = [np.min(self._temp_time_data[TEMPERATURE_KEY]), 
                            np.min(self._temp_time_data2[TEMPERATURE_KEY]), 
                            np.min(self._temp_time_data3[TEMPERATURE_KEY])]
        max_temps = [np.max(self._temp_time_data[TEMPERATURE_KEY]),
                            np.max(self._temp_time_data2[TEMPERATURE_KEY]), 
                            np.max(self._temp_time_data3[TEMPERATURE_KEY])]

        max_index = np.argmax(max_temps)
        temp_func = get_temperature_function_from_data_collection(dc, self._temp_data.state)
        time_values = dc[self._temp_data.state][max_index][TIME_KEY]
        temp_values = dc[self._temp_data.state][max_index][TEMPERATURE_KEY]
        goal_temp_func = np.array([time_values, temp_values]).T
        self.assert_close_arrays(temp_func,goal_temp_func)
