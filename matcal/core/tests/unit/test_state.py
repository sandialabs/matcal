
from copy import copy, deepcopy

from matcal.core.state import State, SolitaryState, StateCollection
from matcal.core.tests.MatcalUnitTest import MatcalUnitTest


class StateTest(MatcalUnitTest):

  def setUp(self) -> None:
    super().setUp(__file__)
    self.st_0 = State("rate_0.001", rate=1e-3, temperature=298.15)
    self.st_1 = State("rate_0.01", rate=1e-2, temperature=298.15)
    self.st_2 = SolitaryState()


  def test_name_property(self):
    self.assertEqual(self.st_0.name, "rate_0.001")

  def test_get_state_parameter(self):
    self.assertAlmostEqual(self.st_0["rate"], 1e-3)
    self.assertAlmostEqual(self.st_0["temperature"], 298.15)
    self.assertIsNone(self.st_2['anything'])

  def test_get_parameters(self):
    self.assertDictEqual(self.st_0.params, {"rate": 1e-3, "temperature": 298.15})

  def test_states_with_same_name_and_parameters_will_be_equal(self):
    dup = State("rate_0.001", rate=1e-3, temperature=298.15)
    self.assertTrue(self.st_0 == dup)
    self.assertFalse(self.st_0 == self.st_1)
    
  def test_empty_and_full_states(self):
    self.assertTrue(self.st_2.solitary_state)
    self.assertFalse(self.st_1.solitary_state)

  def test_states_with_different_name_and_parameters_will_not_be_equal(self):
    dup = State("rate_0.001", rate=1e-3, temperature=298.15)
    self.assertFalse(self.st_0 != dup)
    self.assertTrue(self.st_0 != self.st_1)

  def test_update_state_variable(self):
      self.st_0.update_state_variable("rate", 1e-4)
      self.assertEqual(self.st_0["rate"], 1e-4)

  def test_bad_state_variable_retrieve_raises_key_error(self):
    with self.assertRaises(State.KeyError):
      self.st_0["bad state var"]

  def test_bad_state_variable_name_and_value(self):
    with self.assertRaises(TypeError):
      State("test", state={})

    my_state = State("test")
    my_new_state_vars = {1:10}
    with self.assertRaises(TypeError):
      my_state.update(my_new_state_vars)


class TestStateCollection(MatcalUnitTest):

    def setUp(self):
        super().setUp(__file__)

    @classmethod
    def setUpClass(cls) -> None:
        cls.state1 = State("example1")
        cls.state2 = State("example2")

    def test_createStateCollectionWithOneState(self):
        sc = StateCollection("sc", self.state1)
        self.assertIn(self.state1.name, sc)
        self.assertIs(self.state1, sc[self.state1.name])

    def test_createStateCollectionWithListOfStates(self):
        sc = StateCollection("sc", self.state1, self.state2)
        self.assertIn(self.state1.name, sc)
        self.assertIn(self.state2.name, sc)

    def test_state_collections_equals(self):
        sc = StateCollection("sc", self.state1, self.state2)
        sc2 = copy(sc)
        sc3 = deepcopy(sc)

        state3 = State("3", a=1, b=2)
        state3_2 = State("3", a=1, b=2)

        state3_bad_params = State("3", a=2, b=3)

        state4 = State("4", c=3, d=4)
        state4_2 = State("4", c=3, d=4)

        sc4 = StateCollection("sc4", state3, state4)
        sc4_2 = StateCollection("sc4", state3_2, state4_2)
        sc4_bad_params = StateCollection("sc4", state3_bad_params, state4_2)

        self.assertTrue(sc == sc2)
        self.assertTrue(sc4 == sc4_2)
        self.assertTrue(sc == sc3)

        self.assertTrue(not (sc4 == sc4_bad_params))
