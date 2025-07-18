from collections import OrderedDict
from copy import deepcopy
import numpy as np

from matcal.core.parameters import (MATCAL_PREPROCESSORS, Parameter, ParameterCollection, 
                                    UnitParameterScaler, UserDefinedParameterPreprocessor,  
                                    _convert_serialized_parameter, 
                                    _convert_serialized_parameter_collection, 
                                    serialize_parameter, serialize_parameter_collection)
from matcal.core.tests.MatcalUnitTest import MatcalUnitTest


class ParameterTest(MatcalUnitTest):
  distro_types = ["conTinuous_desigN", "uniform_uncertain"]

  def setUp(self):
    super().setUp(__file__)

  def test_MissingInputs(self):
    self.assert_error_type(TypeError, Parameter)
    self.assert_error_type(TypeError, Parameter, "Name")
    self.assert_error_type(TypeError, Parameter, "Name", 1)

  def test_InappropreateDataInputs(self):
    self.assert_error_type(Parameter.InvalidBoundError, Parameter, "Name", '1', 2)
    self.assert_error_type(Parameter.InvalidBoundError, Parameter, "Name", 2, '3')
    self.assert_error_type(Parameter.InvalidNameError, Parameter, 212, 2, 4)
    self.assert_error_type(Parameter.InvalidBoundError, Parameter, 'Name', [3], 5)
    self.assert_error_type(Parameter.InvalidBoundError, Parameter, "name", 2, [6])
    self.assert_error_type(Parameter.InvalidCurrentValueError, Parameter, "Name", 1, 2, '1.3')
    self.assert_error_type(Parameter.InvalidCurrentValueError, Parameter, "Name", 1, 4, [3])

  def test_InappropreateDataValues(self):
    self.assert_error_type(Parameter.InvalidRangeError, Parameter,"Name", 2, 1)
    self.assert_error_type(Parameter.InvalidRangeError, Parameter,"Name", 1., 5., 6.)
    self.assert_error_type(Parameter.InvalidRangeError, Parameter,"Name", 1, 5, -1)

  def test_DataRetrevial(self):
    p = Parameter("NaMe", 0, 10, 2)
    self.assertAlmostEqual(p.get_lower_bound(), 0)
    self.assertAlmostEqual(p.get_upper_bound(), 10.)
    self.assertAlmostEqual(p.get_current_value(), 2.)
    self.assertTrue(p.get_name() == "NaMe")
    self.assertEqual(p.get_distribution(), "continuous_design")

  def test_defaultInitialValue(self):
    p = Parameter("NaMe", 0, 10)
    self.assertAlmostEqual(p.get_lower_bound(), 0)
    self.assertAlmostEqual(p.get_upper_bound(), 10.)
    self.assertAlmostEqual(p.get_current_value(), 5.)
    self.assertTrue(p.get_name() == "NaMe")

  def test_checkDistrobutionTypes(self):
    p = Parameter("NaMe", 0, 10)
    for d in self.distro_types:
      self.assertIn(d.lower(), p.VALID_DISTRIBUTIONS)

  def test_getDistrTypes(self):

    for d in self.distro_types:
      p = Parameter("name", 1, 2, 1.5, d)
      self.assertEqual(p.get_distribution(), d.lower())

  def test_badDistroType(self):
    with self.assertRaises(Parameter.InvalidDistributionError):
      Parameter("name", 1, 4, 2, "NOT HERE")

  def test_equal(self):
    from copy import deepcopy

    p = Parameter("NaMe", 0, 10)
    p2 = deepcopy(p)
    self.assertEqual(p, p2)
    p2.set_current_value(1)
    self.assertFalse(p==p2) 

  def test_default_preprocessing_tags_is_empty(self):
    p = Parameter('a', 0, 1)
    self.assertEqual(p.preprocessing_tags, [])
    
  def test_add_unit_preprocessing_tag_and_get_it_back(self):
    p = Parameter('a', 0, 1)
    p._add_unit_preprocessing()
    self.assertEqual(p.preprocessing_tags[-1], MATCAL_PREPROCESSORS.unit)
    
  def test_add_log_preprocessing_tag_and_get_it_back(self):
    p = Parameter('a', 0, 1)
    p._add_log_preprocessing()
    self.assertEqual(p.preprocessing_tags[-1], MATCAL_PREPROCESSORS.log)

  def test_log_tag_appears_first(self):
    p = Parameter('a', 0, 1)
    p._add_unit_preprocessing()
    p._add_log_preprocessing()
    self.assertEqual(p.preprocessing_tags[0], MATCAL_PREPROCESSORS.log)
    self.assertEqual(p.preprocessing_tags[1], MATCAL_PREPROCESSORS.unit)


def param_preprocessor_func(params):
  params["new"] = params["1"] + params["4"]
  params["1"] = params["1"] - 1
  return params


class UserDefinedParameterPreprocessorTest(MatcalUnitTest):
  def setUp(self):
    super().setUp(__file__)
    self._parameter_dict = {"1":1, "2": 2, "3":3, "4":4}

  def test_init(self):
    UserDefinedParameterPreprocessor(param_preprocessor_func)

  def test_process_params(self):
    param_preprocessor = UserDefinedParameterPreprocessor(param_preprocessor_func)

    params_updated = param_preprocessor(self._parameter_dict)

    self.assertTrue(params_updated["1"] == 0)
    self.assertTrue(params_updated["new"] == 5)

class UnitParameterScalerTest(MatcalUnitTest):

    def setUp(self):
      super().setUp(__file__)
      
    def test_get_unit_range(self):
      pc = ParameterCollection('test', Parameter('a', 0, 2), Parameter('z', -10, 10), Parameter('w', 10, 10000))
      pc.assign_all_to_unit_preprocessing()
      self._confirm_range_change(pc)

    def _confirm_range_change(self, pc):
        ups = UnitParameterScaler(pc)
        unit_pc = ups.unit_parameter_collection
        self.assertEqual(len(pc), len(unit_pc))
        for param_name in unit_pc:
          low = unit_pc[param_name].get_lower_bound()
          high = unit_pc[param_name].get_upper_bound()
          self.assertAlmostEqual(low, 0)
          self.assertAlmostEqual(high, 1)
        
    def test_get_correct_initial_guess(self):
      pc = ParameterCollection('test', Parameter('a', 0, 2, 1.5), Parameter('b', -10, 10, -10), Parameter('c', -1, 1))
      pc.assign_all_to_unit_preprocessing()
      self._confirm_initial_change(pc, {'a': .75, 'b':0, 'c':.5})

    def test_convert_single_to_unit(self):
      pc = ParameterCollection('new_params', Parameter('A', -10, 10), Parameter('B', 2, 6))
      pc.assign_all_to_unit_preprocessing()
      ups = UnitParameterScaler(pc)
      current_config = {'A':5, 'B': 3}
      goal = {"A":.75, "B":.25}
      test = ups.to_unit_scale(current_config)
      self.assert_close_dicts_or_data(goal, test)
      
    def test_convert_single_from_unit(self):
      pc = ParameterCollection('new_params', Parameter('A', -10, 10), Parameter('B', 2, 6))
      pc.assign_all_to_unit_preprocessing()
      ups = UnitParameterScaler(pc)
      goal = {'A':5, 'B': 3}
      unit_vals = {"A":.75, "B":.25}
      test = ups.from_unit_scale(unit_vals)
      self.assert_close_dicts_or_data(goal, test)      
      
    def test_convert_array_to_unit(self):
      pc = ParameterCollection('new_params', Parameter('A', -10, 10), Parameter('B', 2, 6))
      pc.assign_all_to_unit_preprocessing()
      ups = UnitParameterScaler(pc)
      current_config = {'A':np.array([-10, -5, 0, 5, 10]), 'B': np.array([2, 3, 4, 5, 6])}
      goal = {"A":np.array([0, .25, .5, .75, 1.]), "B":np.array([0, .25, .5, .75, 1.0])}
      test = ups.to_unit_scale(current_config)
      self.assert_close_dicts_or_data(goal, test)

    def test_convert_array_from_unit(self):
      pc = ParameterCollection('new_params', Parameter('A', -10, 10), Parameter('B', 2, 6))
      pc.assign_all_to_unit_preprocessing()
      ups = UnitParameterScaler(pc)
      goal = {'A':np.array([-10, -5, 0, 5, 10]), 'B': np.array([2, 3, 4, 5, 6])}
      unit_vals = {"A":np.array([0, .25, .5, .75, 1.]), "B":np.array([0, .25, .5, .75, 1.0])}
      test = ups.from_unit_scale(unit_vals)
      self.assert_close_dicts_or_data(goal, test)

    def _confirm_initial_change(self, pc, goal_dict):
      ups = UnitParameterScaler(pc)
      unit_pc = ups.unit_parameter_collection
      self.assertEqual(len(pc), len(unit_pc))
      for param_name in unit_pc:
        guess = unit_pc[param_name].get_current_value()
        goal = goal_dict[param_name]
        self.assertAlmostEqual(guess, goal)
        
    def test_do_nothing_if_not_assigned_to(self):
      a = Parameter('a', -3, 6, 0)
      b = Parameter('b', 0, 100)
      b._add_unit_preprocessing()
      pc = ParameterCollection('ab', a, b)
      ups = UnitParameterScaler(pc)
      unit_pc = ups.unit_parameter_collection
      goal_pc = ParameterCollection('ab_goal', Parameter('a', -3, 6, 0), Parameter('b', 0, 1, .5))
      for p_name, goal_param in goal_pc.items():
        test_param = unit_pc[p_name]
        self.assertAlmostEqual(test_param.get_lower_bound(), goal_param.get_lower_bound())
        self.assertAlmostEqual(test_param.get_upper_bound(), goal_param.get_upper_bound())
        self.assertAlmostEqual(test_param.get_current_value(), goal_param.get_current_value())


class ParameterCollectionTest(MatcalUnitTest):

    def setUp(self):
      super().setUp(__file__)

    def test_badInitialize(self):

        with self.assertRaises(ParameterCollection.CollectionTypeError):
            ParameterCollection(123)
            ParameterCollection(['asdf'])

    def test_addViaObject(self):
        q = Parameter("H", 150, 1500, 200)
        PC = ParameterCollection("parameter_collections", q)
        p = Parameter("Y", 50, 500)
        PC.add(p)

        self.assertEqual(PC.get_distribution(), "continuous_design")

        self.assertAlmostEqual(PC["Y"].get_lower_bound(), 50.)
        self.assertAlmostEqual(PC["Y"].get_upper_bound(), 500.)
        self.assertAlmostEqual(PC["Y"].get_current_value(), 275.)

        self.assertAlmostEqual(PC["H"].get_lower_bound(), 150.)
        self.assertAlmostEqual(PC["H"].get_upper_bound(), 1500.)
        self.assertAlmostEqual(PC["H"].get_current_value(), 200.)

    def test_badAddDifferentDist(self):
        q = Parameter("H", 150, 1500, 200, distribution="uniform_uncertain")
        PC = ParameterCollection("parameter_collections", q)
        p = Parameter("Y", 50, 500)
        self.assert_error_type(PC.DifferentDistributionError, PC.add, p)

    def test_retrieveData(self):
        p = Parameter("Y", 50, 500)
        q = Parameter("H", 150, 1500, 200)
        PC = ParameterCollection("parameter_collections", p, q)
        e = Parameter("E", 100, 1000)
        nu = Parameter("nu", 0, .5, .33)
        PC.add(e)
        PC.add(nu)

        self.assertAlmostEqual(PC["Y"].get_lower_bound(), 50.)
        self.assertAlmostEqual(PC["Y"].get_upper_bound(), 500.)
        self.assertAlmostEqual(PC["Y"].get_current_value(), 275.)

        self.assertAlmostEqual(PC["H"].get_lower_bound(), 150.)
        self.assertAlmostEqual(PC["H"].get_upper_bound(), 1500.)
        self.assertAlmostEqual(PC["H"].get_current_value(), 200.)

        self.assertAlmostEqual(PC["E"].get_lower_bound(), 100.)
        self.assertAlmostEqual(PC["E"].get_upper_bound(), 1000.)
        self.assertAlmostEqual(PC["E"].get_current_value(), 550.)

        self.assertAlmostEqual(PC["nu"].get_lower_bound(), 0.)
        self.assertAlmostEqual(PC["nu"].get_upper_bound(), .5)
        self.assertAlmostEqual(PC["nu"].get_current_value(), 0.33)

        self.assertTrue(PC.name == "parameter_collections")

    def test_update_parameter_collection_from_results(self):
        a = Parameter("a", 0, 10)
        b = Parameter("b", -10, 10)
        PC = ParameterCollection("simple", a, b)
        new_pc = {'a':1, 'b':5}
        PC.update_from_results(new_pc)
        self.assertAlmostEqual(PC['a'].get_current_value(), 1)
        self.assertAlmostEqual(PC['b'].get_current_value(), 5)

    def test_update_parameter_collection_from_keyword_value_pairs(self):
        a = Parameter("a", 0, 10)
        b = Parameter("b", -10, 10)
        PC = ParameterCollection("simple", a, b)
        PC.update_parameters(a=1, b=5)
        self.assertAlmostEqual(PC['a'].get_current_value(), 1)
        self.assertAlmostEqual(PC['b'].get_current_value(), 5)


    def test_update_parameter_collection_from_keyword_value_pair_invalid_key(self):
        a = Parameter("a", 0, 10)
        b = Parameter("b", -10, 10)
        PC = ParameterCollection("simple", a, b)
        with self.assertRaises(KeyError):
            PC.update_parameters(a=1, c=5)
        

    def test_CountingData(self):

        e = Parameter("E", 100, 1000)
        nu = Parameter("nu", 0, .5, .33)

        PC = ParameterCollection("parameter_collections")
        names = PC.get_item_names()
        num_param = PC.get_number_of_items()
        self.assertEqual(names, None)
        self.assertEqual(0, num_param)

        PC.add(e)
        names = PC.get_item_names()
        num_param = PC.get_number_of_items()
        self.assertEqual(len(names), 1)
        self.assertEqual(len(names), num_param)
        self.assertEqual(names, ["E"])

        self.assertAlmostEqual(PC["E"].get_lower_bound(), 100.)
        self.assertAlmostEqual(PC["E"].get_upper_bound(), 1000.)
        self.assertAlmostEqual(PC["E"].get_current_value(), 550.)

        PC.add(nu)
        names = PC.get_item_names()
        num_param = PC.get_number_of_items()
        self.assertEqual(len(names), 2)
        self.assertEqual(len(names), num_param)
        self.assertTrue("E" in names)
        self.assertTrue("nu" in names)

    def test_add_nonunique_parameter(self):
        a = Parameter("a", 0, 1, 0.4)
        a1 = Parameter("a", 0, 1, 0.4)

        with self.assertRaises(KeyError):
            collection = ParameterCollection("test", a, a1)

    def test_equal(self):
        a = Parameter("a", 0, 10)
        b = Parameter("b", -10, 10)
        PC = ParameterCollection("simple", a, b)
        PC2 = deepcopy(PC)
        PC == PC2
        self.assertTrue(PC == PC2)
        PC2.update_parameters(a=1)
        self.assertFalse(PC==PC2)
        a = Parameter("a", 1, 10)
        PC2 = ParameterCollection("simple", a, b)
        self.assertFalse(PC==PC2)
        a = Parameter("a", 0, 11)
        PC2 = ParameterCollection("simple", a, b)
        self.assertFalse(PC==PC2)
        a = Parameter("a", 0, 10)
        PC2 = ParameterCollection("simple1", a, b)
        self.assertFalse(PC==PC2)
        a = Parameter("a", 0, 10, units="kg")
        PC2 = ParameterCollection("simple", a, b)
        self.assertFalse(PC==PC2)

        c = Parameter("c", 1, 5, 2)
        PC = ParameterCollection("simple", a, b)
        PC2= ParameterCollection("simple", a, b, c)

        self.assertFalse(PC==PC2)

    def test_add_two_collections(self):
        a = Parameter("a", 0, 10)
        b = Parameter("b", -10, 10)
        c = Parameter("c", 1, 5, 2)
        PC = ParameterCollection("simple", a, b)
        PC2= ParameterCollection("c", c)        

        new_PC = PC+PC2

        self.assertTrue("a" in new_PC.keys())
        self.assertTrue("b" in new_PC.keys())
        self.assertTrue("c" in new_PC.keys())

        self.assertTrue(new_PC.name == (PC.name + " " + PC2.name))
        

    def test_name_update(self):
        a = Parameter("a", 0, 10)
        b = Parameter("b", -10, 10)
        PC = ParameterCollection("simple", a, b)

        self.assertEqual("simple", PC.name)
        PC.set_name("new name")
        self.assertEqual("new name", PC.name)

        with self.assertRaises(ParameterCollection.CollectionTypeError):
            PC.set_name(1)

    def test_return_as_dict(self):
        a = Parameter("a", 0, 10)
        b = Parameter("b", -10, 10)
        PC = ParameterCollection("simple", a, b)

        PC_dict = PC.dict()

        self.assertEqual(OrderedDict, type(PC_dict))
        self.assertEqual(PC_dict["a"], a)
        self.assertEqual(PC_dict["b"], b)

    def test_get_initial_values_as_dict(self):
        a = Parameter("a", 0, 10)
        b = Parameter("b", -10, 10)
        PC = ParameterCollection("simple", a, b)

        initial_value_dict = PC.get_current_value_dict()
        goal = {"a":5, "b":0}

        self.assertEqual(goal, initial_value_dict)

class TestSerializeParameterCollection(MatcalUnitTest):

    def setUp(self):
        super().setUp(__file__)

    def test_serialize_and_convert_parameter(self):
        p_goal = Parameter('a', -1, 2, .3, 'uniform_uncertain', 'um')
        s_param = serialize_parameter(p_goal)
        p_test = _convert_serialized_parameter(s_param)
        self.assertTrue(p_goal == p_test)

    def test_serialize_and_convert_parameter_collection(self):
        p_goals = [Parameter('a', -1, 2, .3, 'continuous_design', 'um')]
        p_goals.append(Parameter('asd', -12, -2, -4, "continuous_design", "miles"))
        pc_goal = ParameterCollection('my_test', *p_goals)
        pc_ser = serialize_parameter_collection(pc_goal)
        pc_test = _convert_serialized_parameter_collection(pc_ser)
        self.assertTrue(pc_test==pc_goal)
        
