from abc import abstractmethod
from collections import OrderedDict
from copy import copy, deepcopy
import matplotlib.pyplot as plt
import numpy as np
import pickle

from matcal.core.data import (convert_dictionary_to_data, 
    Data, DataCollection, \
    scale_data_collection, UnequalTimeDimensionSizeError, 
    _confirm_first_dimension_length, _determine_data_type, 
    convert_data_to_dictionary, _scale_data, RangeDataConditioner, 
    MaxAbsDataConditioner, ReturnPassedDataConditioner, 
    Scaling, ScalingCollection, AverageAbsDataConditioner, 
    DataCollectionStatistics)
from matcal.core.data_importer import FileData
from matcal.core.state import State, SolitaryState
from matcal.core.tests.MatcalUnitTest import MatcalUnitTest


class CommonDataUnitTests(object):
    def __init__():
        pass

    class CommonSetUp(MatcalUnitTest):
        @property
        @abstractmethod
        def _data_class(self):
            """"""
        def setUp(self):
            super().setUp(__file__)

            self.state = State("test")
            self._len = 10
            two_col_array = np.zeros(self._len, dtype=[("x", np.double),("y", np.double)])
            two_col_array["x"] = np.linspace(0, 10, self._len)
            two_col_array["y"] = np.linspace(0.25, 11, self._len)

            self._array = two_col_array
            self._data = self._data_class(self._array)

            self._state = State("my state", rate=1e-4)

    class CommonTests(CommonSetUp):
        def test_invalid_init(self):
            with self.assertRaises(Data.TypeError):
                invalid_name = self._data_class("uh oh bad init")

        def test_bad_state_update(self):
            with self.assertRaises(Data.TypeError):
                self._data.set_state('not a state')

        def test_state_update(self):
            self.assertEqual(self._data.state, SolitaryState())
            self._data.set_state(self._state)
            self.assertEqual(self._data.state, self._state)

        def test_confirm_state(self):
            self._data.set_state(self.state)
            self.assertEqual(self._data.state, self.state)

        def test_bad_key(self):
            with self.assertRaises(ValueError):
                self._data['bad key']

        def test_confirm_field_names(self):
            self.assertTrue("x" in self._data.field_names)
            self.assertTrue("y" in self._data.field_names)

        def test_confirm_length(self):
            self.assertEqual(self._data.length, self._len)

        def test_confirm_length_one(self):
            data_dict = {"x": [1]}
            data = convert_dictionary_to_data(data_dict)
            self.assertEqual(data.length, 1)
            data_dict = {"x": 1}
            data = convert_dictionary_to_data(data_dict)
            self.assertEqual(data.length, 1)

            np.savetxt("test.csv", data, header="x", comments="")
            data = FileData("test.csv")
            self.assertEqual(data.length, 1)

        def test_set_data_name(self):
            self.assertEqual(self._data.name[:9], "data_set_")
            self._data.set_name("test")
            self.assertEqual(self._data.name, "test")

        def test_returns_correct_values(self):
            self.assert_close_arrays(self._data['x'], np.linspace(0,10,self._len))
            self.assert_close_arrays(self._data['y'], np.linspace(.25, 11, self._len))

        def test_data_remove_field(self):

            with self.assertRaises(self._data.KeyError):
                self._data.remove_field("bad_field")

            with self.assertRaises(self._data.TypeError):
                self._data.remove_field(1)

            new_data = self._data.remove_field("y")

            self.assertEqual(new_data.field_names, ['x'])
            with self.assertRaises(ValueError):
                new_data["y"]

        def test_confirm_field_names_no_fields(self):
            self._data = self._data.remove_field("x")
            self._data = self._data.remove_field('y')
            self.assertTrue([] == self._data.field_names)
            
        def test_data_rename_field(self):
            with self.assertRaises(self._data.TypeError):
                self._data.rename_field(1, "X")

            with self.assertRaises(self._data.TypeError):
                new_data = self._data.rename_field("x", 1)

            with self.assertRaises(self._data.KeyError):
                self._data.rename_field("asd", "X")

            self._data.rename_field("y", "Y")
            self.assertEqual(self._data.field_names, ["x", "Y"])
            
            self._data.rename_field("x", "X")
            self.assertEqual(self._data.field_names, ["X", "Y"])
            
        def test_pass_field_data(self):
            n_pts = 5
            two_col_array = np.zeros(self._len, dtype=[("t", np.double),("u", np.double, (n_pts,))])
            t_val = np.linspace(0,100,self._len)
            u_val = np.linspace(0,12, self._len*n_pts).reshape(self._len, n_pts)
            two_col_array['t'] = t_val
            two_col_array['u'] = u_val
            d = self._data_class(two_col_array)
            self.assert_close_arrays(d['t'], t_val)
            self.assert_close_arrays(d['u'], u_val)

        def test_data_is_pickable(self):
            dict_data = {"x":np.linspace(0, 2, 100)}
            dict_data["y"] = dict_data["x"]**2*2+1

            my_data = convert_dictionary_to_data(dict_data)
            my_state = State("the state", a=1, b=2, c=3)
            my_data.set_state(my_state)

            with open("data_pickle.pkl", 'wb') as f:
                pickle.dump(my_data, f)

            with open("data_pickle.pkl", 'rb') as f:
                my_data_2 = pickle.load(f)

            self.assertEqual(my_data.state.name, my_data_2.state.name)
            self.assertEqual(my_data.field_names, my_data_2.field_names)
            self.assertEqual(my_data.length, my_data_2.length)
            self.assertTrue((my_data == my_data_2).all())

        def test_add_field(self):
            dict_data = {"x":np.linspace(0, 2, 100)}
            dict_data["y"] = dict_data["x"]**2*2+1

            my_data = convert_dictionary_to_data(dict_data)
            my_state = State("the state", a=1, b=2, c=3)
            my_data.set_state(my_state)

            new_data  = my_data.add_field("d", np.linspace(0,0.1,100))
            self.assertTrue("d" in new_data.field_names)
            self.assert_close_arrays(new_data["d"], np.linspace(0,0.1,100) )

        def test_add_field_bad_length(self):
            dict_data = {"x":np.linspace(0, 2, 102)}
            dict_data["y"] = dict_data["x"]**2*2+1

            my_data = convert_dictionary_to_data(dict_data)
            my_state = State("the state", a=1, b=2, c=3)
            my_data.set_state(my_state)

            with self.assertRaises(Data.ValueError):
                new_data  = my_data.add_field("d", np.linspace(0,0.1,100))
            
        def test_add_field_bad_name(self):
            dict_data = {"x":np.linspace(0, 2, 100)}
            dict_data["y"] = dict_data["x"]**2*2+1

            my_data = convert_dictionary_to_data(dict_data)
            my_state = State("the state", a=1, b=2, c=3)
            my_data.set_state(my_state)
            with self.assertRaises(Data.TypeError):
                new_data  = my_data.add_field(1, np.linspace(0,0.1,100))


class TestData(CommonDataUnitTests.CommonTests):

    _data_class = Data

class DataCollectionTest(MatcalUnitTest):

    @classmethod
    def setUpClass(cls):
        cls.example_state1 = State("example", val=1, val2=1)
        cls.example_state2 = State("example2", val=1, val2=2)

    def setUp(self):
        super().setUp(__file__)
        self.collection_type = DataCollection
        d0_dict = {"U":[1, 2, 3, 4],"F":[21, 42, 63, 84]}
        self.d_0 = convert_dictionary_to_data(d0_dict)
        self.d_0.set_state(self.example_state1)

        d1_dict = {"U":[1, 2, 3, 4],"F":[4, 8, 12, 16.0]}
        self.d_1 = convert_dictionary_to_data(d1_dict)
        self.d_1.set_state(self.example_state1)

        d2_dict = {"U":[1, 2, 3, 4],"F":[4.4, 8.8, 13.2, 17.6]}
        self.d_2 = convert_dictionary_to_data(d2_dict) 
        self.d_2.set_state(self.example_state2)

        dT_dict = {"U":[1, 2, 3, 4],"T":[300, 310, 321, 329]}
        self.d_t = convert_dictionary_to_data(dT_dict) 
        self.d_t.set_state(self.example_state1)

    def test_invalidName_willRaiseValueError(self):
        with self.assertRaises(DataCollection.CollectionTypeError):
            dc = self.collection_type(None, [])
            cd = self.collection_type(1)

        with self.assertRaises(DataCollection.CollectionValueError):
            dc = self.collection_type("")

    def test_confirm_dumps(self):
        dc = DataCollection('dumper', self.d_0, self.d_1, self.d_2)
        dofs = ["U", "F"]
        dump = dc.dumps()
        state_names = ['example', 'example2']
        self.assert_same_unordered_lists(state_names, list(dump.keys()))
        self.assertEqual(len(dump['example']), 2)
        self.assertEqual(len(dump['example2']),1)
        for dof in dofs:
            self.assert_close_arrays(dump['example'][0][dof], self.d_0[dof])
            self.assert_close_arrays(dump['example'][1][dof], self.d_1[dof])
            self.assert_close_arrays(dump['example2'][0][dof], self.d_2[dof])

        from matcal.core.serializer_wrapper import json_serializer
        json_serializer.dumps(dump) # make sure no error raised

    def test_createDataCollectionWithOneData(self):
        dc = self.collection_type("ex", self.d_0)
        self.assertIn(self.example_state1, dc)
        self.assertIs(dc[self.example_state1][0], self.d_0)

    def test_createDataCollectionWithSameState(self):
        dc = self.collection_type("ex", self.d_0, self.d_1)
        self.assertIn(self.example_state1, dc)
        self.assertIs(dc[self.example_state1][0], self.d_0)
        self.assertIs(dc[self.example_state1][1], self.d_1)

    def test_createDataCollectionWithTwoData(self):
        dc = self.collection_type("ex")
        dc.add([self.d_0, self.d_2])
        self.assertIn(self.example_state1, dc)
        self.assertIn(self.example_state2, dc)
        self.assertIs(dc[self.example_state1][0], self.d_0)
        self.assertIs(dc[self.example_state2][0], self.d_2)

    def test_createDataCollectionWithTwoDataList(self):
        dc = self.collection_type("ex", self.d_0, self.d_2)
        self.assertIn(self.example_state1, dc)
        self.assertIn(self.example_state2, dc)
        self.assertIs(dc[self.example_state1][0], self.d_0)
        self.assertIs(dc[self.example_state2][0], self.d_2)

    def test_data_collection_plot(self):
        plt.close('all')
        dc = self.collection_type("ex", self.d_0, self.d_2)
        dc.plot("U", "F", show=False)
        dc.plot("U", "F", color='k', show=False)
        dc.plot("U", "F", labels="data (#)", show=False)
        dc.plot("U", "F", labels=None, show=False)
        self.assertEqual(len(plt.get_fignums()), 2)

    def test_data_collection_plot_kwargs(self):
        plt.close('all')

        dc = self.collection_type("ex", self.d_0, self.d_2)
        dc.plot("U", "F", color='k', show=False)
        dc.plot("U", "F", color='k', marker='o', show=False)
        dc.plot("U", "F", color='k', marker='o', linestyle='--', show=False)
        self.assertEqual(len(plt.get_fignums()), 2)

    def test_scaled_data_collection_plot(self):
        plt.close('all')

        dc = self.collection_type("ex", self.d_0, self.d_2)
        scaled_dc = scale_data_collection(dc, "T", 10)
        scaled_dc.plot("U", "F", show=False)
        self.assertEqual(len(plt.get_fignums()), 2)

    def test_scaled_data_collection_plot_show_true(self):
        plt.close('all')

        dc = self.collection_type("ex", self.d_0, self.d_2)
        scaled_dc = scale_data_collection(dc, "T", 10)
        scaled_dc.plot("U", "F", block=False)
        self.assertEqual(len(plt.get_fignums()), 2)
        plt.close('all')

    def test_scaled_data_collection_plot_with_bad_field(self):
        plt.close('all')
        dc = self.collection_type("ex", self.d_0, self.d_2)
        scaled_dc = scale_data_collection(dc, "T", 10)
        scaled_dc.plot("U", "F", show=False)
        scaled_dc.plot("U", "bad_field", show=False)
        self.assertEqual(len(plt.get_fignums()), 4)

    def test_data_collection_get_plot_default_label(self):
        dc = self.collection_type("ex", self.d_0, self.d_2)
        default_label = dc._get_default_label("data_name")
        self.assertEqual(default_label, "data_name")
        default_label = dc._get_default_label("/path/to/data")
        self.assertEqual(default_label, "data")
        
    def test_data_collection_get_plot_suppress_label(self):
        dc = self.collection_type("ex", self.d_0, self.d_2)
        label = dc._get_plot_label("suppress", "data_name", 0, 0)
        self.assertEqual(label, "_data_name")
    
    def test_data_collection_get_plot_label_suppress_after_first(self):
        dc = self.collection_type("ex", self.d_0, self.d_2)
        label = dc._get_plot_label("label", "data_name", 1, 1)
        self.assertEqual(label, "_label")

    def test_data_collection_get_plot_numbered_label(self):
        dc = self.collection_type("ex", self.d_0, self.d_2)
        label = dc._get_plot_label("data (#)", "data_name", 0, 0)
        self.assertEqual(label, "data 0")

        default_label = dc._get_plot_label("data (#) yay", "data_name", 0, 0)
        self.assertEqual(default_label, "data 0 yay")

    def test_data_collection_plot_user_fig(self):
        plt.close('all')
        dc = self.collection_type("ex", self.d_0, self.d_2)
        fig = plt.figure("my fig")
        dc.plot("U", "F", figure=fig, show=False)
        self.assertEqual(len(plt.get_fignums()), 1)
        self.assertTrue("my fig" in plt.get_figlabels())

    def test_data_collection_plot_bad_user_fig(self):
        plt.close('all')
        dc = self.collection_type("ex", self.d_0, self.d_2)
        with self.assertRaises(dc.CollectionTypeError):
            dc.plot("U", "F", figure="not a fig", show=False)

    def test_data_collection_plot_one_state(self):
        plt.close('all')
        dc = self.collection_type("ex", self.d_0, self.d_2)
        dc.plot("U", "F", state=self.d_0.state, show=False)
        self.assertEqual(len(plt.get_fignums()), 1)
        self.assertTrue(self.d_0.state.name+" U F" in plt.get_figlabels())

    def test_access_data_with_state_name(self):
        dc = self.collection_type("ex", self.d_0, self.d_2)
        self.assertIn(self.example_state1, dc)
        self.assertIn(self.example_state2, dc)
        self.assertIs(dc[self.example_state1.name][0], self.d_0)
        self.assertIs(dc[self.example_state2.name][0], self.d_2)

    def test_createHeterogeneousDataCollection(self):
        dc = self.collection_type("Mixed", self.d_0, self.d_t)
        self.assertIn("T", dc.field_names)
        self.assertIn("U", dc.field_names)
        self.assertIn("F", dc.field_names)

    def test_equals(self):
        state_slow = State("slow", rate=1.0)
        state_fast = State("fast", rate=2.0)

        d0_dict = {"U":[1., 2, 3, 4],"F":np.array([21, 42, 63., 84])}
        d_0 = convert_dictionary_to_data(d0_dict)
        d_0.set_state(state_fast)
        
        d1_dict = {"U":[1., 2, 3, 4],"F":np.array([4, 8, 12, 16.])}
        d_1 = convert_dictionary_to_data(d1_dict)
        d_1.set_state(state_fast)

        d2_dict = {"U":[1., 2, 3, 4],"F":np.array([4.4, 8.8, 13.2, 17.6])}
        d_2 = convert_dictionary_to_data(d2_dict) 
        d_2.set_state(state_slow)

        d3_dict = {"U":[1., 2, 3, 4],"T":np.array([12., 24, 36, 48])}
        d_3 = convert_dictionary_to_data(d3_dict) 
        d_3.set_state(state_slow)

        data_1 = DataCollection("test", d_0, d_1, d_2, d_3)
        data_1_copy = copy(data_1)
        data_1_deep_copy = deepcopy(data_1)
        data_2 = data_1 + data_1_copy
        data_2_copy = copy(data_2)

        self.assertTrue(data_1 == data_1)
        self.assertTrue(data_1 == data_1_copy)
        self.assertTrue(data_1 == data_1_deep_copy)

        self.assertTrue(data_2 == data_2_copy)
        self.assertFalse(data_1 == data_2)

    def test_not_equal_different_states(self):
        dc = self.collection_type("ex", self.d_0, self.d_1, self.d_2)
        dc2 = self.collection_type("ex", self.d_0, self.d_1)
        dc3 = self.collection_type("Mixed", self.d_0, self.d_t)

        self.assertFalse(dc == dc2)
        self.assertFalse(dc == dc3)
        
    def test_set_name(self):
        dc = self.collection_type("ex", self.d_0, self.d_1, self.d_2)
        self.assertEqual(dc.name, "ex")

        dc.set_name("new name")
        self.assertEqual(dc.name, "new name")

    def test_access_by_state_and_state_name(self):
        dc = self.collection_type("ex", self.d_0, self.d_1, self.d_2)
        state1_data_by_state_name = dc[self.example_state1.name]
        state1_data_by_state = dc[self.example_state1]

        self.assertEqual(state1_data_by_state_name, state1_data_by_state)
        with self.assertRaises(KeyError):
            dc["invalid state name"]
        
        with self.assertRaises(KeyError):
            dc[SolitaryState()]

        with self.assertRaises(KeyError):
            dc[1]

    def test_raise_error_with_non_unique_state_names(self):
        state0 = State("MYNAME", a=1)
        state1 = State("MYNAME", a=2)

        self.d_0.set_state(state0)
        self.d_1.set_state(state1)

        dc = DataCollection('my_collection')
        dc.add(self.d_0)
        with self.assertRaises(DataCollection.NonUniqueStateNameError):
            dc.add(self.d_1)

    def test_remove_field(self):
        dc = self.collection_type("Mixed", self.d_0, self.d_1, self.d_2, self.d_t)
        dc.remove_field("F")
        self.assertEqual(dc.field_names, ["U", "T"])
        with self.assertRaises(dc.CollectionTypeError):
            dc.remove_field(1)
        dc.remove_field("invalid field")

    def test_state_field_names(self):
        dc = self.collection_type("Mixed", self.d_0, self.d_1, self.d_2, self.d_t)
        state1_field_names_name_key = dc.state_field_names("example")
        state1_field_names_state_key = dc.state_field_names(self.example_state1)

        state2_field_names_name_key = dc.state_field_names("example2")
        state2_field_names_state_key = dc.state_field_names(self.example_state2)

        state1_goal = ["T", "F", "U"]
        state2_goal = ["F", "U"]


        self.assertEqual(state1_field_names_name_key, state1_field_names_state_key)
        self.assertEqual(state2_field_names_name_key, state2_field_names_state_key)

        for entry in state1_goal:
            self.assertTrue(entry in state1_field_names_state_key)
        
        for entry in state2_goal:
            self.assertTrue(entry in state2_field_names_state_key)

    def test_state_field_names(self):
        dc = self.collection_type("Mixed", self.d_0, self.d_1, self.d_2, self.d_t)
        state1_field_names_name_key = dc.state_common_field_names("example")
        state1_field_names_state_key = dc.state_common_field_names(self.example_state1)

        state2_field_names_name_key = dc.state_common_field_names("example2")
        state2_field_names_state_key = dc.state_common_field_names(self.example_state2)

        state1_goal = ["U", ]
        state2_goal = ["F", "U"]


        self.assertEqual(state1_field_names_name_key, state1_field_names_state_key)
        self.assertEqual(state2_field_names_name_key, state2_field_names_state_key)

        for entry in state1_goal:
            self.assertTrue(entry in state1_field_names_state_key)
        
        for entry in state2_goal:
            self.assertTrue(entry in state2_field_names_state_key)

    def test_select_data_by_state_values(self):
        dc = self.collection_type("Mixed", self.d_0, self.d_1, 
                                  self.d_2, self.d_t)
        sub_dc = dc.get_data_by_state_values(val=1)
        self.assertEqual(dc, sub_dc)

        sub_dc = dc.get_data_by_state_values(val=1, val2=1)

        self.assertNotEqual(dc, sub_dc)

        self.assertTrue(self.d_0.state in sub_dc)

        self.assert_list_contains_array(sub_dc[self.d_0.state], self.d_0)

        self.assertTrue(self.d_1.state in sub_dc)
        self.assert_list_contains_array(sub_dc[self.d_1.state], self.d_1)

        self.assertTrue(self.d_t.state in sub_dc)
        self.assert_list_contains_array(sub_dc[self.d_t.state], self.d_t)

        self.assertTrue(self.d_2.state not in sub_dc)
        sub_dc = dc.get_data_by_state_values(val=1, val2=2)
        self.assertTrue(self.d_2.state in sub_dc)
        self.assert_list_contains_array(sub_dc[self.d_2.state], self.d_2)
        self.assertTrue(self.d_0.state not in sub_dc)
        self.assertTrue(self.d_1.state not in sub_dc)
        self.assertTrue(self.d_t.state not in sub_dc)

        sub_dc = dc.get_data_by_state_values(val3=1, val1=20)
        self.assertTrue(sub_dc.state_names == [])

    def test_select_data_by_state_values_get_name(self):
        dc = self.collection_type("Mixed", self.d_0, self.d_1, 
                                  self.d_2, self.d_t)
        sub_dc = dc.get_data_by_state_values(val=1, val2=2)
        self.assertTrue(sub_dc.name == "Mixed_with_state_params_val_1_val2_2")

    def test_get_states_by_state_values(self):
        dc = self.collection_type("Mixed", self.d_0, self.d_1, 
                                  self.d_2, self.d_t)
        sub_sc = dc.get_states_by_state_values(val=1)
        self.assertEqual(dc.states, sub_sc)

        sub_sc = dc.get_states_by_state_values(val=1, val2=1)
        self.assertNotEqual(dc.states, sub_sc)
        self.assertTrue(self.d_0.state in sub_sc.values())
        self.assertTrue(self.d_1.state in sub_sc.values())
        self.assertTrue(self.d_t.state in sub_sc.values())
        self.assertTrue(self.d_2.state not in sub_sc.values())

        sub_sc = dc.get_states_by_state_values(val=1, val2=2)
        self.assertTrue(self.d_2.state in sub_sc.values())
        self.assertTrue(self.d_0.state not in sub_sc.values())
        self.assertTrue(self.d_1.state not in sub_sc.values())
        self.assertTrue(self.d_t.state not in sub_sc.values())

    def test_has_report_statistics_method(self):
        p_means = {'a': 2, 'b': 10}
        p_sd = {'a':1, 'b': 2}
        n_sets = 10
        
        def toy_fun(a, b, idx):
            x = np.linspace(-1, 1)
            y = a * np.power(x, 1) + b 
            return {'x':x, "y": y}
        
        dc = self._generate_dc_for_stats(p_means, p_sd, toy_fun, n_sets)
        
        dc.report_statistics('x')

    def test_report_statistics_raises_error_if_no_indep_field(self):
        p_means = {'a': 2, 'b': 10}
        p_sd = {'a':1, 'b': 2}
        n_sets = 10
        
        def toy_fun(a, b, idx):
            x = np.linspace(-1, 1)
            y = a * np.power(x, 1) + b 
            return {'x':x, "y": y}
        
        dc = self._generate_dc_for_stats(p_means, p_sd, toy_fun, n_sets)
        
        with self.assertRaises(TypeError):
            dc.report_statistics()

    def test_has_report_statistics_reports_interp_locations(self):
        p_means = {'a': 2, 'b': 10}
        p_sd = {'a':1, 'b': 2}
        n_sets = 10
        
        def toy_fun(a, b, idx):
            x = np.linspace(-1, 1)
            y = a * np.power(x, 1) + b 
            return {'x':x, "y": y}
        
        dc = self._generate_dc_for_stats(p_means, p_sd, toy_fun, n_sets)
        
        report = dc.report_statistics('x')        
        ref = toy_fun(1, 1, 0)
        self.assert_close_arrays(report['matcal_default_state']['locations'], ref['x'])

    def test_has_report_statistics_reports_interp_locations_across_min_and_max_all_data(self):
        p_means = {'a': 2, 'b': 10}
        p_sd = {'a':1, 'b': 2}
        n_sets = 11
        
        def toy_fun(a, b, idx):
            n_points = np.random.randint(20,40)
            x = np.linspace(-1 + idx/10, 1+idx/10, n_points)
            y = a * np.power(x, 1) + b 
            return {'x':x, "y": y}
        
        dc = self._generate_dc_for_stats(p_means, p_sd, toy_fun, n_sets)
        
        report = dc.report_statistics('x')
        n_expected_points = self._get_num_expected_pts(dc, list(dc.states.keys())[0])
        goal_x = np.linspace(-1, 2, n_expected_points)
        self.assert_close_arrays(report['matcal_default_state']['locations'], goal_x)

    def _get_num_expected_pts(self, dc, state):
        dc_stats = DataCollectionStatistics()
        n_expected_points = dc_stats._get_number_of_field_points('x', dc[state])
        return n_expected_points

    def test_report_statistics_has_all_dependent_fields(self):
        p_means = {'a': 2, 'b': 10}
        p_sd = {'a':1, 'b': 2}
        n_sets = 11
        
        def toy_fun(a, b, idx):
            n_points = np.random.randint(20,40)
            x = np.linspace(-1 + idx/10, 1+idx/10, n_points)
            y = a * np.power(x, 1) + b 
            return {'x':x, "y": y, 'c':np.ones_like(y)}
        
        dc = self._generate_dc_for_stats(p_means, p_sd, toy_fun, n_sets)
        
        ref = toy_fun(1,1,1)
        indep_var = 'x'
        report = dc.report_statistics(indep_var)   
        
        for field_name in ref:
            if field_name != indep_var:
                self.assertIn(field_name, report['matcal_default_state'].keys())
            
    def test_costant_field_has_near_zero_sd(self):
        p_means = {'a': 2, 'b': 10}
        p_sd = {'a':1, 'b': 1}
        n_sets = 11
        
        def toy_fun(a, b, idx):
            n_points = np.random.randint(20,40)
            x = np.linspace(-1 + idx/10, 1+idx/10, n_points)
            y = a * np.power(x, 1) + b 
            y_noise = np.random.normal(0, .1)
            return {'x':x, "y": y, 'c':np.ones_like(y)}
        
        dc = self._generate_dc_for_stats(p_means, p_sd, toy_fun, n_sets)
        
        ref = toy_fun(1,1,1)
        indep_var = 'x'
        report = dc.report_statistics(indep_var)   
        
        n_expected_points = self._get_num_expected_pts(dc, list(dc.states.keys())[0])

        const_data = report['matcal_default_state']['c']
        self.assert_close_arrays(const_data['std dev'], np.zeros(n_expected_points)) 
        self.assert_close_arrays(const_data['mean'], np.ones(n_expected_points))
    
    def test_recover_gauss_noise(self):
        p_means = {'a': 2, 'b': 0}
        p_sd = {'a':1e-8, 'b': 1e-8}
        n_sets = 10000
    
        goal_sd = .1
        
        def toy_fun(a, b, idx):
            n_points = np.random.randint(20,40)
            x = np.linspace(-1 , 1, n_points)
            y = a * x + b 
            y_noise = np.random.normal(0, goal_sd)
            return {'x':x, "y": y+y_noise, 'c':np.ones_like(y)}
        
        dc = self._generate_dc_for_stats(p_means, p_sd, toy_fun, n_sets)
        
        indep_var = 'x'
        report = dc.report_statistics(indep_var)   
        
        n_expected_points = self._get_num_expected_pts(dc, list(dc.states.keys())[0])

        noise_data = report['matcal_default_state']['y']
        x = report['matcal_default_state']['locations']
        self.assert_close_arrays(noise_data['mean'], p_means['a'] * x + p_means['b'],
                                 show_on_fail=True, atol = 1e-2)
        self.assert_close_arrays(noise_data['std dev'], goal_sd * np.ones(n_expected_points),
                                 show_on_fail=True, atol = 1e-2) 

    def _generate_dc_for_stats(self, p_means, p_sd, toy_fun, n_sets):
        p_sets = {}
        for name in p_means:
            p_sets[name] = np.random.normal(p_means[name], p_sd[name], n_sets)
        data_instances = []
        for i_set in range(n_sets):
            a = p_sets['a'][i_set]
            b = p_sets['b'][i_set]
            data_dict = toy_fun(a, b, i_set)
            data_instances.append(convert_dictionary_to_data(data_dict))
        
        dc = DataCollection('test', *data_instances)
        return dc


class TestDictionaryToData(MatcalUnitTest):
    def setUp(self):
        super().setUp(__file__)
        self.simple_data_dict = {"time": [1, 2, 3], "values": [10, 20, 30]}
        self.simple_data = self._make_simple_data()

    def _make_simple_data(self):
        simple_data = convert_dictionary_to_data(self.simple_data_dict)
        return simple_data

    def test_get_correct_default_state(self):
        self.assertEqual(self.simple_data.state.name, SolitaryState().name)

    def test_recreate_simple_data(self):
        for fname in self.simple_data.field_names:
            self.assertTrue(np.allclose(self.simple_data[fname], self.simple_data_dict[fname]))

    def test_read_in_dict(self):
        dict_data = {"A": [1, 2, 3], "Beta": np.ones(3)}
        convert_dictionary_to_data(dict_data)

    def test_produce_single_entry_data(self):
        dict_data = {"only": 2}
        data = convert_dictionary_to_data(dict_data)

        self.assertTrue(data['only'] == dict_data['only'])

    def test_fail_on_none_entry(self):
        dict_data = {"a": [1, 2], 'b': None}
        with self.assertRaises(Data.TypeError):
            convert_dictionary_to_data(dict_data)

    def test_produce_multiple_entries_same_length(self):
        dict_data = {"A": [1, 2, 3], "Beta": np.ones(3)}
        data = convert_dictionary_to_data(dict_data)
        for key in dict_data.keys():
            self.assertTrue(np.array_equal(data[key], dict_data[key]))

    def test_fail_entries_different_length(self):
        dict_data = {"A": [1, 2, 3], "Beta": np.ones(3), "cat": [1]}
        with self.assertRaises(UnequalTimeDimensionSizeError):
            data = convert_dictionary_to_data(dict_data)
        dict_data = {"A": [[1, 2, 3],[0,0,0],[3,3,3]], "Beta": np.ones([3,5]), "cat": [1]}
        with self.assertRaises(UnequalTimeDimensionSizeError):
            data = convert_dictionary_to_data(dict_data)

    def test_determine_data_type_1d_returns_touple_wo_size(self):
        n = 5
        data = np.zeros(n)
        key = 'fake'
        data_type = _determine_data_type(data, key)
        self.assertEqual(len(data_type), 2)
        self.assertEqual(data_type[0], key)
        self.assertEqual(data_type[1], 'float')

    def test_determine_data_type_2d_returns_touple_w_size(self):
        n = 5
        m = 10
        data = np.zeros((n,m))
        key = 'fake'
        data_type = _determine_data_type(data, key)
        self.assertEqual(len(data_type), 3)
        self.assertEqual(data_type[0], key)
        self.assertEqual(data_type[1], 'float')
        self.assertEqual(data_type[2], (m,))

    def test_converting_to_dict_then_to_data(self):
        goal_dict = {"A": [1, 2, 3], "B": [3, 4, 5], "C": [-1, -2, -3]}
        data = convert_dictionary_to_data({"A": [1, 2, 3], "B": [3, 4, 5]})
        data_dict = {key: data[key] for key in data.field_names}
        data_dict["C"] = [-1, -2, -3]
        conv_data = convert_dictionary_to_data(data_dict)
        goal_data = convert_dictionary_to_data(goal_dict)
        for field in goal_data.field_names:
            self.assert_close_arrays(conv_data[field], goal_data[field])
        self.assertListEqual(conv_data.field_names, goal_data.field_names)

    def test_convert_different_dimensions(self):
        goal_a = [3,2,1]
        goal_m = [[-1,-2],[10,20],[50, 25]]
        dict_data = {"A": goal_a, "M": goal_m}
        data = convert_dictionary_to_data(dict_data)
        m = data['M']
        a = data['A']
        self.assert_close_arrays(a, goal_a)
        self.assert_close_arrays(m, goal_m)

    def test_confirm_dimension_length(self):
        ref_n = 3
        self.assertTrue(_confirm_first_dimension_length(ref_n, (3,)))
        self.assertTrue(_confirm_first_dimension_length(ref_n, (3,2)))
        self.assertTrue(_confirm_first_dimension_length(ref_n, (3,5,8)))
        self.assertTrue(_confirm_first_dimension_length(ref_n*2, (6,4)))

        self.assertFalse(_confirm_first_dimension_length(5, (2,)))
        self.assertFalse(_confirm_first_dimension_length(5, (1,2)))
        self.assertFalse(_confirm_first_dimension_length(5, (1,5)))


class TestDataToDictionary(MatcalUnitTest):

    def setUp(self):
        super().setUp(__file__)
        self.simple_dict = {"A": np.zeros(4), "B": np.array([1, 2, 3, 4])}
        self.list_dict = {"A": [0,0,0,0], "B": [1, 2, 3, 4]}
        self.matcal_dict = OrderedDict(self.simple_dict)
        self.simple_data = convert_dictionary_to_data(self.simple_dict)
        self.list_data = convert_dictionary_to_data(self.list_dict)
        self.matcal_data = convert_dictionary_to_data(self.matcal_dict)

    def test_return_dictionary_from_data(self):
        self.assertIsInstance(convert_data_to_dictionary(self.simple_data), dict)
        self.assertIsInstance(convert_data_to_dictionary(self.matcal_data), dict)

    def test_get_input_dict_from_simple_dict(self):
        self._confirm_same_dict(self.simple_data, self.simple_dict)

    def test_get_input_dict_from_matcal_dict(self):
        self._confirm_same_dict(self.matcal_data, self.matcal_dict)

    def test_get_input_dict_from_list_dict(self):
        self._confirm_same_dict(self.list_data, self.list_dict)

    def _confirm_same_dict(self, data, my_dict):
        return_dict = convert_data_to_dictionary(data)
        for key in my_dict.keys():
            self.assert_close_arrays(return_dict[key], my_dict[key])

    def test_bad_call_to_convert_data_to_dict(self):
        with self.assertRaises(TypeError):
            return_dict = convert_data_to_dictionary("invalid")
        

def get_conditioner_test_data():
    simple_data_dict = {"time":np.linspace(0, 1, 11), "value":np.linspace(-4,6,11)}
    simple_data = convert_dictionary_to_data(simple_data_dict)

    simple_data_dict2 = {"time":np.linspace(0.4, 0.8, 5), "value":np.linspace(2,4,5)}
    simple_data2 = convert_dictionary_to_data(simple_data_dict2)

    simple_data_dict3 = {"time":np.linspace(0,1,11), "value":np.linspace(0,10,11)}
    simple_data3 = convert_dictionary_to_data(simple_data_dict3)

    complex_data_dict = {"time":np.linspace(0,1,11), "value":np.linspace(0,10,11), 
                        "zero_range":np.ones(11)*10, 
                        "erratic":[2, -2, -100, 1e-3, 78, 12, 44, -18, 3.14159827, -1e-1, 2], 
                        "zeros":np.zeros(11)}
    complex_data = convert_dictionary_to_data(complex_data_dict)
    return simple_data, simple_data2, simple_data3, complex_data

def get_conditioner_test_data_with_noise():
    data1_dict = {"time":np.linspace(0,1,11), "value":np.linspace(0,10,11), 
                        "value_noise":np.ones(11)*0.5}
    data1 = convert_dictionary_to_data(data1_dict)
    
    data2_dict = {"time":np.linspace(0,1,11), "value":np.linspace(-2,18,11), 
                        "value_noise":np.ones(11)*0.25}
    data2 = convert_dictionary_to_data(data2_dict)
    return data1, data2


class TestScaledDataFunction(MatcalUnitTest):

    def setUp(self):
        super().setUp(__file__)
        data_tup = get_conditioner_test_data()
        self._complex_data = data_tup[3]

    def test_take_in_scaling_collection_and_data(self):
        sc = ScalingCollection("model")
        _scale_data(sc, self._complex_data)

    def test_return_data(self):
        sc = ScalingCollection("model")
        new_data = _scale_data(sc, self._complex_data)
        self.assertIsInstance(new_data, Data)

    def test_return_same_data(self):
        sc = ScalingCollection("model")
        s_data = _scale_data(sc, self._complex_data)
        old_data = self._complex_data
        for key in old_data.field_names:
            self.assertTrue(np.allclose(s_data[key], old_data[key]))

    def test_return_doubled_value(self):
        sc = ScalingCollection("model", Scaling("value", 2))
        s_data = _scale_data(sc, self._complex_data)
        old_data = self._complex_data
        for key in old_data.field_names:
            if key == "value":
                mult = 2
            else:
                mult = 1
            self.assertTrue(np.allclose(s_data[key], mult*old_data[key]))

    def test_return_doubled_plus_1_value(self):
        sc = ScalingCollection("model", Scaling("value", 2, .5))
        s_data = _scale_data(sc, self._complex_data)
        old_data = self._complex_data
        for key in old_data.field_names:
            if key == "value":
                mult = 2
                disp = 1
            else:
                mult = 1
                disp = 0
            self.assertTrue(np.allclose(s_data[key], mult*old_data[key] + disp))

    def test_return_correct_state(self):
        sc = ScalingCollection("model", Scaling("value", 2))
        s_data = _scale_data(sc, self._complex_data)
        old_data = self._complex_data
        self.assertEqual(old_data.state.name, s_data.state.name)        


class ConditionerBaseTests:
    class CommonTests(MatcalUnitTest):
        @abstractmethod
        def conditioner_class(self):
            """"""

        def setUp(self):
            super().setUp(__file__)
            path = self.get_current_files_path(__file__)
            data_tup = get_conditioner_test_data()
            self._simple_data = data_tup[2]
            self._simple_data2 = data_tup[1]
            self._simple_data3 = data_tup[0]
            self._complex_data = data_tup[3]

        @abstractmethod
        def get_scale_and_offset(*args):
            """"""

        def test_raise_non_data_type_error(self):
            dc = self.conditioner_class()
            with self.assertRaises(TypeError):
                dc.apply_to_data(None)
            with self.assertRaises(TypeError):
                dc.apply_to_data("string")
            with self.assertRaises(TypeError):
                dc.apply_to_data({"A": [1, 2, 3]})

        def test_return_a_data_class(self):
            dc = self.conditioner_class()
            dc.initialize_data_conditioning_values([self._simple_data])
            cond_data = dc.apply_to_data(self._simple_data)
            self.assertIsInstance(cond_data, Data)

        def test_returned_conditioned_data(self):
            dc = self.conditioner_class()
            dc.initialize_data_conditioning_values([self._simple_data])
            cond_data = dc.apply_to_data(self._simple_data)

            scale_time, offset_time = self.get_scale_and_offset(self._simple_data["time"])
            scale_value, offset_value = self.get_scale_and_offset(self._simple_data["value"])
            
            goal_time = (self._simple_data["time"]-offset_time)/scale_time
            goal_value = (self._simple_data["value"]-offset_value)/scale_value
            self.assertTrue(np.allclose(goal_time, cond_data['time']))
            self.assertTrue(np.allclose(goal_value, 
                                        cond_data["value"]))

        def test_return_complex_data_with_a_zero_range(self):
            dc = self.conditioner_class()
            dc.initialize_data_conditioning_values([self._complex_data])
            cond_data = dc.apply_to_data(self._complex_data)
                        
            scale_time, offset_time = self.get_scale_and_offset(self._complex_data["time"])
            scale_value, offset_value = self.get_scale_and_offset(self._complex_data["value"])
            scale_zero, offset_zero = self.get_scale_and_offset(self._complex_data["zeros"])
            scale_zero_range, offset_zero_range = self.get_scale_and_offset(
                self._complex_data["zero_range"])
            
            goal_time = (self._complex_data["time"]-offset_time)/scale_time
            goal_value = (self._complex_data["value"]-offset_value)/scale_value
            goal_zero = (self._complex_data["zeros"]-offset_zero)/scale_zero
            goal_zero_range = (self._complex_data["zero_range"]-offset_zero_range)/scale_zero_range

            self.assertTrue(np.allclose(goal_time, cond_data['time']))
            self.assertTrue(np.allclose(goal_value, 
                                        cond_data["value"]))
            self.assertTrue(np.allclose(goal_zero, cond_data['zeros']))
            self.assertTrue(np.allclose(goal_zero_range, 
                                        cond_data["zero_range"]))

        def test_return_condition_dat_with_multi_data_init(self):
            dc = self.conditioner_class()
            dc.initialize_data_conditioning_values([self._complex_data, self._simple_data])

            cond_data = dc.apply_to_data(self._complex_data)
            scale_time, offset_time = self.get_scale_and_offset(self._simple_data["time"] ,
                                                                self._complex_data["time"])
            scale_value, offset_value = self.get_scale_and_offset(self._simple_data["value"], 
                                                                  self._complex_data["value"])
            scale_zero, offset_zero = self.get_scale_and_offset(self._complex_data["zeros"])
            scale_zero_range, offset_zero_range = self.get_scale_and_offset(
                self._complex_data["zero_range"])
          
            goal_time = (self._complex_data["time"]-offset_time)/scale_time
            goal_value = (self._complex_data["value"]-offset_value)/scale_value
            goal_zero = (self._complex_data["zeros"]-offset_zero)/scale_zero
            goal_zero_range = (self._complex_data["zero_range"]-offset_zero_range)/scale_zero_range

            self.assertTrue(np.allclose(goal_time, cond_data['time']))
            self.assertTrue(np.allclose(goal_value, 
                                        cond_data["value"]))
            self.assertTrue(np.allclose(goal_zero, cond_data['zeros']))
            self.assertTrue(np.allclose(goal_zero_range, 
                                        cond_data["zero_range"]))

        def test_apply_first_conditioning_to_new_data(self):
            dc = self.conditioner_class()
            dc.initialize_data_conditioning_values([self._simple_data])
            
            scale_time, offset_time = self.get_scale_and_offset(self._simple_data["time"])
            scale_value, offset_value = self.get_scale_and_offset(self._simple_data["value"])            

            cond_data = dc.apply_to_data(self._simple_data2)
            goal_time = (self._simple_data2["time"]-offset_time)/scale_time
            goal_value = (self._simple_data2["value"]-offset_value)/scale_value

            self.assertTrue(np.allclose(goal_time,
                                        cond_data['time']))
            self.assertTrue(np.allclose(goal_value,
                                        cond_data["value"]))

        def test_apply_first_conditioning_to_new_data_and_noise(self):
            data1, data2 = get_conditioner_test_data_with_noise()
            dc = self.conditioner_class()
            dc.initialize_data_conditioning_values([data1, data2])

            cond_data = dc.apply_to_data(data2)
            scale_value, offset_value = self.get_scale_and_offset(data1['value'], data2['value'])        

            self.assertTrue(np.allclose((data2['value']-offset_value)/scale_value,
                                         cond_data["value"]))
            self.assertTrue(np.allclose(data2['value_noise']/scale_value, cond_data['value_noise']))

        def test_apply_first_conditioning_to_new_data_and_noise_missing_noise_conditioning_field(self):
            data1, data2 = get_conditioner_test_data_with_noise()
            dc = self.conditioner_class()
            dc.initialize_data_conditioning_values([data1, data2])

            data_dict = {"time":np.linspace(0,10,10), "unique":np.random.uniform(0,1,10), "unique_noise":np.ones(10)}
            data_with_noise_field_not_in_conditioner = convert_dictionary_to_data(data_dict)

            cond_data = dc.apply_to_data(data_with_noise_field_not_in_conditioner)
            for field in cond_data.field_names:
                if field in data1.field_names+data2.field_names:
                    scale, offset = self.get_scale_and_offset(data1[field], data2[field])
                else:
                    scale = 1.0
                    offset = 0.0
                self.assertTrue(np.allclose(cond_data[field], 
                                            (data_with_noise_field_not_in_conditioner[field]
                                             -offset)/scale))

        def test_condition_initialize_empty_list_errors(self):
            dc = self.conditioner_class()
            with self.assertRaises(ValueError):
                dc.initialize_data_conditioning_values([])
           
        def test_condition_data_conditioner_not_initialized(self):
            self._simple_data
            dc = self.conditioner_class()
            with self.assertRaises(RuntimeError):
                dc.apply_to_data(self._simple_data)



class TestReturnPassedDataConditioner(ConditionerBaseTests.CommonTests):
    conditioner_class = ReturnPassedDataConditioner
    def get_scale_and_offset(self, *args):
        return 1.0, 0.0


class TestRangeDataConditioner(ConditionerBaseTests.CommonTests):
    conditioner_class = RangeDataConditioner
    def get_scale_and_offset(self, *args):
        if len(args) > 1:
            combined_data = np.concatenate(args)
        else:
            combined_data = args[0]
        scale = np.max(combined_data)-np.min(combined_data)
        if np.abs(scale) < 1e-14:
            scale = np.max(np.abs(combined_data))
        if np.abs(scale) < 1e-14:
            scale = 1.0
        return scale, np.min(combined_data)
    

class TestMaxAbsConditioner(ConditionerBaseTests.CommonTests):
    conditioner_class = MaxAbsDataConditioner
    def get_scale_and_offset(self, *args):
        if len(args) > 1:
            combined_data = np.concatenate(args)
        else:
            combined_data = args[0]
        scale = np.max(np.abs(combined_data))
        if scale < 1e-14:
            scale =1 
        return scale, 0.0


class TestAverageAbsConditioner(ConditionerBaseTests.CommonTests):
    conditioner_class = AverageAbsDataConditioner
    def get_scale_and_offset(self, *args):
        if len(args) > 1:
            combined_data = np.concatenate(args)
        else:
            combined_data = args[0]
        scale = np.average(np.abs(combined_data))
        if scale < 1e-14:
            scale =1.0 
        return scale, 0.0


class ScalingTest(MatcalUnitTest):

    def setUp(self):

        super().setUp(__file__)
        y = {"dependent variable": np.array([-1, 0, 1, 2]),
             "independent variable": np.array([0, 0.33, 0.66, 1])}
        self.data = convert_dictionary_to_data(y)

    def test_init(self):
        s = Scaling("load")
        s = Scaling("load", scalar=10)
        s = Scaling("load", scalar=10.02)
        s = Scaling("load", scalar=10.02, offset=1)

    def test_bad_init(self):

        with self.assertRaises(Scaling.ScalingTypeError):
            Scaling(123, 'asd')
        with self.assertRaises(Scaling.ScalingTypeError):
            Scaling("as", scalar='1')

    def test_properties(self):
        s = Scaling("field_name")
        self.assertEqual(s.field, "field_name")
        self.assertEqual(s.scalar, 1)
        self.assertEqual(s.offset, 0)

    def test_no_scale(self):
        s = Scaling("dependent variable")
        data_new = s.apply_to_data(self.data)
        self.assertAlmostEqual(np.sum(np.abs(np.array(data_new["dependent variable"]) -
                                             self.data["dependent variable"])),
                               0, delta=1e-10)

    def test_scalar_scale(self):
        s = Scaling("dependent variable",  scalar=-5)
        ygoal = self.data["dependent variable"]*-5
        ynew = s.apply_to_data(self.data)
        self.assertAlmostEqual(np.sum(np.abs(np.array(ynew["dependent variable"]) - ygoal)), 0, delta=1e-10)

    def test_scalar_scale_invalid_fields(self):
        s = Scaling("no field of this name", scalar=-5)
        with self.assertRaises(ValueError):
            ynew = s.apply_to_data(self.data)

    def test_scale(self):
        s = Scaling("dependent variable", scalar=-5)
        ygoal = np.array([5, 0, -5, -10])
        ynew = s.apply_to_data(self.data)
        self.assertAlmostEqual(np.sum(np.abs(np.array(ynew["dependent variable"]) - ygoal)), 0,
                               delta=1e-10)

    def test_offset(self):
        s = Scaling("dependent variable", offset=-5)
        ygoal = np.array([-6, -5, -4, -3])
        ynew = s.apply_to_data(self.data)
        self.assertAlmostEqual(np.sum(np.abs(np.array(ynew["dependent variable"]) - ygoal)), 0,
                               delta=1e-10)


class TestScaledDataFunction(MatcalUnitTest):

    def setUp(self):
        super().setUp(__file__)
        data_tup = get_conditioner_test_data()
        self._complex_data = data_tup[3]

    def test_take_in_scaling_collection_and_data(self):
        sc = ScalingCollection("model")
        _scale_data(sc, self._complex_data)

    def test_return_data(self):
        sc = ScalingCollection("model")
        new_data = _scale_data(sc, self._complex_data)
        self.assertIsInstance(new_data, Data)

    def test_return_same_data(self):
        sc = ScalingCollection("model")
        s_data = _scale_data(sc, self._complex_data)
        old_data = self._complex_data
        for key in old_data.field_names:
            self.assertTrue(np.allclose(s_data[key], old_data[key]))

    def test_return_doubled_value(self):
        sc = ScalingCollection("model", Scaling("value", 2))
        s_data = _scale_data(sc, self._complex_data)
        old_data = self._complex_data
        for key in old_data.field_names:
            if key == "value":
                mult = 2
            else:
                mult = 1
            self.assertTrue(np.allclose(s_data[key], mult*old_data[key]))

    def test_return_doubled_plus_1_value(self):
        sc = ScalingCollection("model", Scaling("value", 2, .5))
        s_data = _scale_data(sc, self._complex_data)
        old_data = self._complex_data
        for key in old_data.field_names:
            if key == "value":
                mult = 2
                disp = 1
            else:
                mult = 1
                disp = 0
            self.assertTrue(np.allclose(s_data[key], mult*old_data[key] + disp))

    def test_return_correct_state(self):
        sc = ScalingCollection("model", Scaling("value", 2))
        s_data = _scale_data(sc, self._complex_data)
        old_data = self._complex_data
        self.assertEqual(old_data.state.name, s_data.state.name)

class TestScaleDataCollection(MatcalUnitTest):

    def setUp(self):
        super().setUp(__file__)
        data_tup = get_conditioner_test_data()
        self._complex_data = data_tup[3]
        self._state = State("test")
        self._complex_data.set_state(self._state)
        self._data_collection = DataCollection("test", self._complex_data)

    def test_inputs_and_return(self):
        new_data_collection = scale_data_collection(self._data_collection, "value", 2)
        self.assertIsInstance(new_data_collection, DataCollection)

    def test_bad_inputs_and_raise_error(self):
        with self.assertRaises(Data.TypeError):
            new_data_collection = scale_data_collection("not a dc", "value", 2)
        with self.assertRaises(Data.TypeError):
            new_data_collection = scale_data_collection(self._data_collection, 1, 2)
        with self.assertRaises(Data.TypeError):
            new_data_collection = scale_data_collection(self._data_collection, "value", 'val')
        with self.assertRaises(Data.TypeError):
            new_data_collection = scale_data_collection(self._data_collection, "value", 1,"1")

    def test_scale_results(self):
        new_data_collection = scale_data_collection(self._data_collection, "value", 2)
        new_data = new_data_collection[self._state][0]
        old_data = self._complex_data
        for field in old_data.field_names:
            if field == 'value':
                mult = 2
            else:
                mult = 1
            self.assertTrue(np.allclose(new_data[field], mult*old_data[field]))

    def test_offset_result(self):
        new_data_collection = scale_data_collection(self._data_collection, "value", 1, -1)
        new_data = new_data_collection[self._state][0]
        old_data = self._complex_data
        for field in old_data.field_names:
            if field == 'value':
                offset = -1
            else:
                offset = 0
            self.assertTrue(np.allclose(new_data[field], old_data[field]+offset))

    def test_offset_and_scale_result(self):
        new_data_collection = scale_data_collection(self._data_collection, "value", 2, -1)
        new_data = new_data_collection[self._state][0]
        old_data = self._complex_data
        for field in old_data.field_names:
            if field == 'value':
                offset = -1
                mult = 2
            else:
                offset = 0
                mult = 1
            self.assertTrue(np.allclose(new_data[field], mult*(old_data[field]+offset)))


class ScalingCollectionTest(MatcalUnitTest):

    def setUp(self):
        self.s_0 = Scaling("displacement", 2)
        self.s_1 = Scaling("load", 4)
        self.s_2 = Scaling("time", 60)
        super().setUp(__file__)

    def test_invalidName_willRaiseValueError(self):
        with self.assertRaises(ScalingCollection.CollectionValueError):
            sc = ScalingCollection("")

    def test_invalidType(self):
        with self.assertRaises(ScalingCollection.CollectionTypeError):
            sc = ScalingCollection(None, [])
            sc = ScalingCollection(1)

    def test_createScalingCollectionWithOneScale(self):
        sc = ScalingCollection("test", self.s_0)
        self.assertIn(self.s_0.field, sc)

    def test_createScalingCollectionWithMultipleScales(self):
        sc = ScalingCollection("test", self.s_0, self.s_1, self.s_2)
        self.assertIn(self.s_0.field, sc)
        self.assertIn(self.s_1.field, sc)
        self.assertIn(self.s_2.field, sc)

    def test_scalling_collection_add(self):
        sc1 = ScalingCollection('test', self.s_0)
        sc2 = ScalingCollection('test2', self.s_1, self.s_2)

        sc_full = sc1 + sc2

        self.assertIn(self.s_0.field, sc_full)
        self.assertIn(self.s_1.field, sc_full)
        self.assertIn(self.s_2.field, sc_full)
        self.assertIn(self.s_0, sc_full[self.s_0.field])
        self.assertIn(self.s_1, sc_full[self.s_1.field])
        self.assertIn(self.s_2, sc_full[self.s_2.field])

        self.assertEqual(sc_full.name, sc1.name + " " + sc2.name)


class TestDataCollectionStatistics(MatcalUnitTest):

    def setUp(self):

        super().setUp(__file__)

    def test_set_number_of_interpolation_points(self):
        data1 = convert_dictionary_to_data({"indep":np.linspace(0,1,100), 
                                            "dep":np.linspace(0,10, 100)})
        data2 = convert_dictionary_to_data({"indep":np.linspace(0,1,300), 
                                            "dep":np.linspace(0,10, 300)})
        dc = DataCollection("test", data1, data2)
        stats = DataCollectionStatistics()
        n_pts = stats._get_number_of_field_points("indep", dc[SolitaryState()])
        self.assertEqual(n_pts, 200)
        stats.set_number_of_interpolation_points(5)
        n_pts = stats._get_number_of_field_points("indep", dc[SolitaryState()])
        self.assertEqual(n_pts, 5)

    def test_set_percentiles(self):
        stats = DataCollectionStatistics()
        self.assertEqual(stats._percentiles, [])
        stats.set_percentiles_to_evaluate(0, 1, 2, 98, 99, 100)
        self.assertEqual(stats._percentiles, [0, 1,2,98,99, 100])
        with self.assertRaises(ValueError):
            stats.set_percentiles_to_evaluate(-1)
        with self.assertRaises(TypeError):
            stats.set_percentiles_to_evaluate('wrong type')
    
    def test_generate_statistics_default(self):
        data1 = convert_dictionary_to_data({"indep":np.linspace(0,1,100), 
                                            "dep":np.linspace(0,10, 100)})
        data2 = convert_dictionary_to_data({"indep":np.linspace(0,1,300), 
                                            "dep":np.linspace(0,10, 300)})
        dc = DataCollection("test", data1, data2)
        stats = DataCollectionStatistics()
        stats_results = stats.generate_state_statistics("indep", dc, 
            SolitaryState())
        self.assert_close_arrays(np.linspace(0,10, 200), 
                                 stats_results['dep']['mean'])
        self.assert_close_arrays(np.zeros(200), 
                                 stats_results['dep']['std dev'])
        data1 = convert_dictionary_to_data({"indep":np.linspace(0,1,100), 
                                            "dep":np.linspace(0,20, 100)})
        data2 = convert_dictionary_to_data({"indep":np.linspace(0,1,300), 
                                            "dep":np.linspace(0,40, 300)})
        dc = DataCollection("test", data1, data2)
        stats_results = stats.generate_state_statistics("indep", dc, 
            SolitaryState())
        self.assert_close_arrays(np.linspace(0,30, 200), 
                                 stats_results['dep']['mean'])
        self.assert_close_arrays(np.linspace(0,1,200)*np.std([20,40]), 
                                 stats_results['dep']['std dev'])
        
    def test_generate_statistics_default(self):
        stats = DataCollectionStatistics()
        stats.set_percentiles_to_evaluate(0,100)
        data1 = convert_dictionary_to_data({"indep":np.linspace(0,1,100), 
                                            "dep":np.linspace(0,20, 100)})
        data2 = convert_dictionary_to_data({"indep":np.linspace(0,1,300), 
                                            "dep":np.linspace(0,40, 300)})
        dc = DataCollection("test", data1, data2)
        with self.assertRaises(KeyError):
            stats_results = stats.generate_state_statistics("bad_field", dc, 
                SolitaryState())
        bad_state = State("bad state")
        with self.assertRaises(KeyError):
            stats_results = stats.generate_state_statistics("indep", dc, 
                bad_state)

        stats_results = stats.generate_state_statistics("indep", dc, 
            SolitaryState())
        self.assert_close_arrays(np.linspace(0,30, 200), 
                                 stats_results['dep']['mean'])
        self.assert_close_arrays(np.linspace(0,1,200)*np.std([20,40]), 
                                 stats_results['dep']['std dev'])
        self.assert_close_arrays(np.linspace(0,20,200), stats_results['dep']["percentile_0"])
        self.assert_close_arrays(np.linspace(0,40,200), stats_results['dep']["percentile_100"])

    def test_interpolator_change(self):
        from scipy.interpolate import make_interp_spline
        stats = DataCollectionStatistics(interpolation_tool=make_interp_spline, k=3)
        self.assertEqual(stats._interpolation_tool, make_interp_spline)
        self.assertEqual(stats._interpolation_kwargs, {"k":3})

        stats.set_percentiles_to_evaluate(0,100)
        data1 = convert_dictionary_to_data({"indep":np.linspace(0,1,100), 
                                            "dep":np.linspace(0,20, 100)})
        data2 = convert_dictionary_to_data({"indep":np.linspace(0,1,300), 
                                            "dep":np.linspace(0,40, 300)})
        dc = DataCollection("test", data1, data2)
        stats_results = stats.generate_state_statistics("indep", dc, 
            SolitaryState())
        self.assert_close_arrays(np.linspace(0,30, 200), 
                                 stats_results['dep']['mean'])
        self.assert_close_arrays(np.linspace(0,1,200)*np.std([20,40]), 
                                 stats_results['dep']['std dev'])
        self.assert_close_arrays(np.linspace(0,20,200), stats_results['dep']["percentile_0"])
        self.assert_close_arrays(np.linspace(0,40,200), stats_results['dep']["percentile_100"])

    def test_set_sort_ascending(self):
        from scipy.interpolate import make_interp_spline
        stats = DataCollectionStatistics(interpolation_tool=make_interp_spline)
        self.assertTrue(stats._sort_ascending)
        stats.set_sort_ascending(False)
        self.assertFalse(stats._sort_ascending)

        data1 = convert_dictionary_to_data({"indep":[1.0, 0.5, 2.0, 3.0, 1.5], 
                                            "dep":[0, 1, 2, 3, 4]})
        data2 = convert_dictionary_to_data({"indep":np.linspace(0,1,300), 
                                            "dep":np.linspace(0,40, 300)})
        dc = DataCollection("test", data1, data2)
        with self.assertRaises(RuntimeError):
            stats.generate_state_statistics("indep", dc, SolitaryState())
        stats.set_sort_ascending()
        self.assertTrue(stats._sort_ascending)
        res = stats.generate_state_statistics("indep", dc, SolitaryState())
        self.assertIn("dep", res)
        
    def test_interpolator_err(self):
        def bad_interp_func():
            return
        
        stats = DataCollectionStatistics(interpolation_tool=bad_interp_func)
        
        stats.set_percentiles_to_evaluate(0,100)
        data1 = convert_dictionary_to_data({"indep":np.linspace(0,1,100), 
                                            "dep":np.linspace(0,20, 100)})
        data2 = convert_dictionary_to_data({"indep":np.linspace(0,1,300), 
                                            "dep":np.linspace(0,40, 300)})
        dc = DataCollection("test", data1, data2)
        with self.assertRaises(RuntimeError):
            stats_results = stats.generate_state_statistics("indep", dc, 
                SolitaryState())
