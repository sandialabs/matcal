import numpy as np

from matcal.core.data import DataCollection, convert_dictionary_to_data, RangeDataConditioner
from matcal.core.objective import CurveBasedInterpolatedObjective, \
                                  L2NormMetricFunction, L1NormMetricFunction, \
                                  ObjectiveCollection, ObjectiveSet, \
                                  _normalize_residuals, \
                                  NormMetricFunction
from matcal.core.objective_results import ObjectiveResults, flatten_data_collection
from matcal.core.state import SolitaryState, State 

from matcal.core.tests.MatcalUnitTest import MatcalUnitTest


class TestObjectiveResults(MatcalUnitTest):

    def setUp(self):
        super().setUp(__file__)

    def test_getObjectiveValue(self):
        r = ObjectiveResults(["x"], ["x"])
        data_dict = {"x":0}
        state=SolitaryState()
        data = convert_dictionary_to_data(data_dict)
        data.set_state(state)
        r.add_weighted_conditioned_objective(data)
        self.assertEqual(r.objectives[state][0]["x"], 0)

    def test_assign_large_data_sets(self):
        r = ObjectiveResults(["x"], ["x"])
        self.assertFalse(r._large_data_sets)
        r = ObjectiveResults(["x"], ["x"], True)
        self.assertTrue(r._large_data_sets)

    def test_getResidualValue(self):
        r = ObjectiveResults(["x"], ["x"])
        data_dict = {"x":[0, 1, 2]}
        state=SolitaryState()
        data = convert_dictionary_to_data(data_dict)
        data.set_state(state)
        r.add_weighted_conditioned_residuals(data)

        for i in range(3):
            self.assertEqual(r.weighted_conditioned_residuals[state][0]["x"][i], i)

    def test_twoStateObjectives(self):
        r = ObjectiveResults(["x"], ["x"])
        data_dict = {"x":-1}
        state=SolitaryState()
        data = convert_dictionary_to_data(data_dict)
        data.set_state(state)
        r.add_weighted_conditioned_objective(data)

        data_dict = {"x":5}
        state2=State("my state")
        data2 = convert_dictionary_to_data(data_dict)
        data2.set_state(state2)
        r.add_weighted_conditioned_objective(data2)

        self.assertEqual(r.objectives[state][0]["x"][0], -1)
        self.assertEqual(r.objectives[state2][0]["x"][0], 5)

    def test_twoStateResiduals(self):
        r = ObjectiveResults(["x"], ["x"])

        data_dict = {"x":[0, 1, 2]}
        state=SolitaryState()
        data = convert_dictionary_to_data(data_dict)
        data.set_state(state)
        r.add_weighted_conditioned_residuals(data)

        data_dict = {"x":[0, -1, -2]}
        state2=State("my state")
        data2 = convert_dictionary_to_data(data_dict)
        data2.set_state(state2)
        r.add_weighted_conditioned_residuals(data2)

        for i in range(3):
            self.assertEqual(r.weighted_conditioned_residuals[state][0]["x"][i], i)
            self.assertEqual(r.weighted_conditioned_residuals[state2][0]["x"][i], -i)

    def test_getAllResiduals(self):
        r = ObjectiveResults(["x"], ["x"])

        data_dict = {"x":[0, 1, 2]}
        state=SolitaryState()
        data = convert_dictionary_to_data(data_dict)
        data.set_state(state)
        r.add_weighted_conditioned_residuals(data)

        data_dict = {"x":[0, -1, -2]}
        state2=State("my state")
        data2 = convert_dictionary_to_data(data_dict)
        data2.set_state(state2)
        r.add_weighted_conditioned_residuals(data2)
        res = r.get_flattened_weighted_conditioned_residuals()
        for i in range(3):
            self.assertEqual(res[i], i)
            self.assertEqual(res[i + 3], -i)

    def test_get_normalized_residuals_one_state(self):
        r = ObjectiveResults(["x"], ["x"])
        n_data = 2
        n_states = 1
        state=SolitaryState()

        a = np.array([0.0,1,2])
        goal_a, data = self._make_norm_data(n_data, n_states, state, a)
        r.add_weighted_conditioned_residuals(data)

        b = np.array([0.0, -2, -4, -6, -8])
        goal_b, data2 = self._make_norm_data(n_data, n_states, state, b)
        r.add_weighted_conditioned_residuals(data2)

        res = _normalize_residuals(r._weighted_conditioned_residuals, 
            L2NormMetricFunction().normalize_only, 
            L2NormMetricFunction().calculate_group_normalization_factor)
        res = flatten_data_collection(res, ['x'])
        goal  = np.concatenate([goal_a, goal_b])
        self.assert_close_arrays(res, goal)

    def test_get_normalized_residuals_two_states(self):
        r = ObjectiveResults(["x"], ["x"])
        n_data1 = 2
        n_data2 = 1
        n_states = 2
        state1 = State("s1")
        state2 = State("s2")

        a = np.array([0.0,1.0,2.0])
        goal_a, data = self._make_norm_data(n_data1, n_states, state1, a)
        r.add_weighted_conditioned_residuals(data)

        b = np.array([0.0, -2.0, -4.0, -6.0, -8.0])
        goal_b, data2 = self._make_norm_data(n_data1, n_states, state1, b)
        r.add_weighted_conditioned_residuals(data2)

        c = np.array([10.0,20.0,30.0, 40.0])
        goal_c, data3 = self._make_norm_data(n_data2, n_states, state2, c)
        r.add_weighted_conditioned_residuals(data3)

        res = _normalize_residuals(r._weighted_conditioned_residuals, 
            L2NormMetricFunction().normalize_only, 
            L2NormMetricFunction().calculate_group_normalization_factor)
        res = flatten_data_collection(res, ['x'])
        goal  = np.concatenate([goal_a, goal_b, goal_c])
        self.assert_close_arrays(res, goal, show_on_fail=True)

    def test_confirm_residual_for_model_set_has_norm_of_one(self):
        r = ObjectiveResults(["x"], ["x"])
        n_data1 = 2
        n_data2 = 1
        n_states = 2
        state1 = State("s1")
        state2 = State("s2")

        a = np.ones(5)
        goal_a, data = self._make_norm_data(n_data1, n_states, state1, a)
        r.add_weighted_conditioned_residuals(data)

        b = np.ones(20)
        goal_b, data2 = self._make_norm_data(n_data1, n_states, state1, b)
        r.add_weighted_conditioned_residuals(data2)

        c = np.ones(100)
        goal_c, data3 = self._make_norm_data(n_data2, n_states, state2, c)
        r.add_weighted_conditioned_residuals(data3)
        
        res = _normalize_residuals(r._weighted_conditioned_residuals, 
            L2NormMetricFunction().normalize_only, 
            L2NormMetricFunction().calculate_group_normalization_factor)
        res = flatten_data_collection(res, ['x'])
        self.assertAlmostEqual(np.linalg.norm(res), 1.)

    def test_confirm_residual_for_model_set_has_norm_of_one_l1_norm(self):
        r = ObjectiveResults(["x"], ["x"])
        n_data1 = 2
        n_data2 = 1
        n_states = 2
        state1 = State("s1")
        state2 = State("s2")

        a = np.ones(5)
        goal_a, data = self._make_norm_data(n_data1, n_states, state1, a)
        r.add_weighted_conditioned_residuals(data)

        b = np.ones(20)
        goal_b, data2 = self._make_norm_data(n_data1, n_states, state1, b)
        r.add_weighted_conditioned_residuals(data2)

        c = np.ones(100)
        goal_c, data3 = self._make_norm_data(n_data2, n_states, state2, c)
        r.add_weighted_conditioned_residuals(data3)
        
        res = _normalize_residuals(r._weighted_conditioned_residuals, 
            L1NormMetricFunction().normalize_only, 
            L1NormMetricFunction().calculate_group_normalization_factor)
        res = flatten_data_collection(res, ['x'])
        self.assertAlmostEqual(np.linalg.norm(res,1), 1.)

    def test_confirm_residual_for_model_set_has_norm_of_one_l0_norm(self):
        r = ObjectiveResults(["x"], ["x"])
        n_data1 = 2
        n_data2 = 1
        n_states = 2
        state1 = State("s1")
        state2 = State("s2")

        a = np.ones(5)
        goal_a, data = self._make_norm_data(n_data1, n_states, state1, a)
        r.add_weighted_conditioned_residuals(data)

        b = np.ones(20)
        goal_b, data2 = self._make_norm_data(n_data1, n_states, state1, b)
        r.add_weighted_conditioned_residuals(data2)

        c = np.ones(100)
        goal_c, data3 = self._make_norm_data(n_data2, n_states, state2, c)
        r.add_weighted_conditioned_residuals(data3)
        
        metric_func = NormMetricFunction(0)

        res = _normalize_residuals(r._weighted_conditioned_residuals, metric_func.normalize_only, 
            metric_func.calculate_group_normalization_factor)
        self.assertAlmostEqual(np.linalg.norm(flatten_data_collection(res), 0), 125.0)

    def test_confirm_residual_for_model_set_has_norm_of_one_l_inf_norm(self):
        r = ObjectiveResults(["x"], ["x"])
        n_data1 = 2
        n_data2 = 1
        n_states = 2
        state1 = State("s1")
        state2 = State("s2")

        a = np.ones(5)
        goal_a, data = self._make_norm_data(n_data1, n_states, state1, a)
        r.add_weighted_conditioned_residuals(data)

        b = np.ones(20)
        goal_b, data2 = self._make_norm_data(n_data1, n_states, state1, b)
        r.add_weighted_conditioned_residuals(data2)

        c = np.ones(100)
        goal_c, data3 = self._make_norm_data(n_data2, n_states, state2, c)
        r.add_weighted_conditioned_residuals(data3)
        
        metric_func = NormMetricFunction(np.inf)

        res = _normalize_residuals(r._weighted_conditioned_residuals, 
            metric_func.normalize_only, metric_func.calculate_group_normalization_factor)
        res = flatten_data_collection(res, ['x'])
        self.assertAlmostEqual(np.linalg.norm(res, np.inf), 1.0)

    def test_confirm_residual_for_model_set_has_norm_of_one_l_minus_inf_norm(self):
        r = ObjectiveResults(["x"], ["x"])
        n_data1 = 2
        n_data2 = 1
        n_states = 2
        state1 = State("s1")
        state2 = State("s2")

        a = np.ones(5)
        goal_a, data = self._make_norm_data(n_data1, n_states, state1, a)
        r.add_weighted_conditioned_residuals(data)

        b = np.ones(20)
        goal_b, data2 = self._make_norm_data(n_data1, n_states, state1, b)
        r.add_weighted_conditioned_residuals(data2)

        c = np.ones(100)
        goal_c, data3 = self._make_norm_data(n_data2, n_states, state2, c)
        r.add_weighted_conditioned_residuals(data3)
        
        metric_func = NormMetricFunction(-np.inf)

        res = _normalize_residuals(r._weighted_conditioned_residuals, 
            metric_func.normalize_only, 
            metric_func.calculate_group_normalization_factor)
        res = flatten_data_collection(res, ["x"])
        self.assertAlmostEqual(np.linalg.norm(res, -np.inf), 1.0)

    def test_confirm_residual_for_model_set_has_norm_of_one_l_minus_2_norm(self):
        r = ObjectiveResults(["x"], ["x"])
        n_data1 = 2
        n_data2 = 1
        n_states = 2
        state1 = State("s1")
        state2 = State("s2")

        a = np.ones(5)
        goal_a, data = self._make_norm_data(n_data1, n_states, state1, a)
        r.add_weighted_conditioned_residuals(data)

        b = np.ones(20)
        goal_b, data2 = self._make_norm_data(n_data1, n_states, state1, b)
        r.add_weighted_conditioned_residuals(data2)

        c = np.ones(100)
        goal_c, data3 = self._make_norm_data(n_data2, n_states, state2, c)
        r.add_weighted_conditioned_residuals(data3)
        
        metric_func = NormMetricFunction(-2)

        res = _normalize_residuals(r._weighted_conditioned_residuals, 
            metric_func.normalize_only, metric_func.calculate_group_normalization_factor)
        res = flatten_data_collection(res, ['x'])
        self.assertAlmostEqual(np.linalg.norm(res, -2), 1.0)

    def test_confirm_residual_for_model_set_has_norm_of_one_l_minus_1_norm(self):
        r = ObjectiveResults(["x"], ["x"])
        n_data1 = 2
        n_data2 = 1
        n_states = 2
        state1 = State("s1")
        state2 = State("s2")

        a = np.ones(5)
        goal_a, data = self._make_norm_data(n_data1, n_states, state1, a)
        r.add_weighted_conditioned_residuals(data)

        b = np.ones(20)
        goal_b, data2 = self._make_norm_data(n_data1, n_states, state1, b)
        r.add_weighted_conditioned_residuals(data2)

        c = np.ones(100)
        goal_c, data3 = self._make_norm_data(n_data2, n_states, state2, c)
        r.add_weighted_conditioned_residuals(data3)
        
        metric_func = NormMetricFunction(-1)

        res = _normalize_residuals(r._weighted_conditioned_residuals, 
            metric_func.normalize_only, metric_func.calculate_group_normalization_factor)
        res = flatten_data_collection(res, ['x'])
        self.assertAlmostEqual(np.linalg.norm(res, -1), 1.0)

    def test_confirm_non_one_residual_for_model_set_has_norm_of_one_l_minus_1_norm(self):
        r = ObjectiveResults(["x"], ["x"])
        n_data1 = 2
        n_data2 = 1
        n_states = 2
        state1 = State("s1")
        state2 = State("s2")

        a = np.ones(5)*2
        goal_a, data = self._make_norm_data(n_data1, n_states, state1, a)
        r.add_weighted_conditioned_residuals(data)

        b = np.ones(20)*2
        goal_b, data2 = self._make_norm_data(n_data1, n_states, state1, b)
        r.add_weighted_conditioned_residuals(data2)

        c = np.ones(100)*2
        goal_c, data3 = self._make_norm_data(n_data2, n_states, state2, c)
        r.add_weighted_conditioned_residuals(data3)
        
        metric_func = NormMetricFunction(-1)

        res = _normalize_residuals(r._weighted_conditioned_residuals, metric_func.normalize_only, 
            metric_func.calculate_group_normalization_factor)
        res = flatten_data_collection(res)
        self.assertAlmostEqual(np.linalg.norm(res, -1), 2.0)

    def test_confirm_non_one_residual_for_model_set_has_norm_of_one_l_minus_2_norm(self):
        r = ObjectiveResults(["x"], ["x"])
        n_data1 = 2
        n_data2 = 1
        n_states = 2
        state1 = State("s1")
        state2 = State("s2")

        a = np.ones(5)*2
        goal_a, data = self._make_norm_data(n_data1, n_states, state1, a)
        r.add_weighted_conditioned_residuals(data)

        b = np.ones(20)*2
        goal_b, data2 = self._make_norm_data(n_data1, n_states, state1, b)
        r.add_weighted_conditioned_residuals(data2)

        c = np.ones(100)*2
        goal_c, data3 = self._make_norm_data(n_data2, n_states, state2, c)
        r.add_weighted_conditioned_residuals(data3)
        
        metric_func = NormMetricFunction(-2)

        res = _normalize_residuals(r._weighted_conditioned_residuals, 
            metric_func.normalize_only, metric_func.calculate_group_normalization_factor)
        res = flatten_data_collection(res, ['x'])

        self.assertAlmostEqual(np.linalg.norm(res, -2), 2.0)

    def test_confirm_residual_is_weighted_between_uneven_states(self):
        r = ObjectiveResults(["x"], ["x"])
        n_data1 = 2
        n_data2 = 1
        n_states = 2
        state1 = State("s1")
        state2 = State("s2")

        a = np.ones(5)
        goal_a, data = self._make_norm_data(n_data1, n_states, state1, a)
        r.add_weighted_conditioned_residuals(data)

        b = np.ones(20)
        goal_b, data2 = self._make_norm_data(n_data1, n_states, state1, b)
        r.add_weighted_conditioned_residuals(data2)

        c = np.ones(100)
        goal_c, data3 = self._make_norm_data(n_data2, n_states, state2, c)
        r.add_weighted_conditioned_residuals(data3)

        res = _normalize_residuals(r._weighted_conditioned_residuals,
            L2NormMetricFunction().normalize_only,
            L2NormMetricFunction().calculate_group_normalization_factor)
        s1_r = np.concatenate((res["s1"][0], res["s1"][0]))
        s2_r = res["s2"][0]
        self.assertAlmostEqual(np.linalg.norm(s1_r), np.linalg.norm(s2_r))

    def test_confirm_residual_is_weighted_between_uneven_data_sizes(self):
        r = ObjectiveResults(["x"], ["x"])
        n_data1 = 2
        n_data2 = 1
        n_states = 2
        state1 = State("s1")
        state2 = State("s2")

        a = np.ones(5)
        goal_a, data = self._make_norm_data(n_data1, n_states, state1, a)
        r.add_weighted_conditioned_residuals(data)

        b = np.ones(20)
        goal_b, data2 = self._make_norm_data(n_data1, n_states, state1, b)
        r.add_weighted_conditioned_residuals(data2)

        c = np.ones(100)
        goal_c, data3 = self._make_norm_data(n_data2, n_states, state2, c)
        r.add_weighted_conditioned_residuals(data3)

        res = _normalize_residuals(r._weighted_conditioned_residuals, 
            L2NormMetricFunction().normalize_only, 
            L2NormMetricFunction().calculate_group_normalization_factor)
        d1_r = res["s1"][0]
        d2_r = res["s1"][1]
        self.assertAlmostEqual(np.linalg.norm(d1_r), np.linalg.norm(d2_r))

    def _make_norm_data(self, n_data, n_states, state, a):
        goal_a = a / np.sqrt(len(a) * n_data * n_states)
        data_dict = {"x":a}
        data = convert_dictionary_to_data(data_dict)
        data.set_state(state)
        return goal_a,data

    def test_calibration_results(self):
        r = ObjectiveResults(['x'],['x'])
        res = np.random.uniform(0,1,10)
        self.assertIsNone(r.calibration_residuals)
        data = convert_dictionary_to_data({"x":res})
        dc = DataCollection("test", data)
        r.set_weighted_conditioned_normalized_residuals(dc)
        self.assert_close_arrays(res, r.calibration_residuals)

    def test_raise_error_when_improper_array_passed(self):
        self._confirm_raises_invalid_residual_error(None)
        self._confirm_raises_invalid_residual_error('a')
        self._confirm_raises_invalid_residual_error([1,2,3])
        self._confirm_raises_invalid_residual_error(np.zeros([2,3]))
        self._confirm_raises_invalid_residual_error(np.ones(8).reshape(-1,1))
       
    def _confirm_raises_invalid_residual_error(self, bad_res):
        r = ObjectiveResults(['x'],['x'])
        with self.assertRaises(ObjectiveResults.InvalidResidualError):
            r.set_weighted_conditioned_normalized_residuals(bad_res)

    def test_inspect_populated_results(self):
        short_dc, long_dc, sim_qois_dc, cond_short_dc, cond_long_dc,  cond_sim_qois_dc, results, qois = self._make_extensive_results()

        self.assertEqual(len(qois.get_flattened_weighted_conditioned_experiment_qois()), 10)
        self.assertEqual(len(qois.get_flattened_weighted_conditioned_simulation_qois()), 10)
        self.assertEqual(len(results.get_flattened_weighted_conditioned_residuals()), 10)
        self.assertEqual(len(results.get_flattened_weighted_conditioned_objectives()), 2)

        self.assertEqual(cond_short_dc, qois.conditioned_experiment_data)
        self.assertEqual(short_dc, qois.experiment_data)
        short_dc.set_name("qois ref")
        self.assertEqual(short_dc, qois.experiment_qois)

        self.assertEqual(cond_long_dc, qois.conditioned_simulation_data)
        self.assertEqual(long_dc, qois.simulation_data)

        self.assertEqual(cond_sim_qois_dc, qois.conditioned_simulation_qois)
        self.assertEqual(sim_qois_dc, qois.simulation_qois)

        for i in range(len(results.residuals["one"][0]["Y"])):
            resid = i/4*(2.0-3.0)
            self.assertAlmostEqual(results.residuals["one"][0]["Y"][i], resid)
            self.assertAlmostEqual(results.conditioned_residuals["one"][0]['Y'][i], resid/3)
     
        for i in range(len(results.residuals["two"][0]["Y"])):
            resid = i/4*(3.0-2.0)
            self.assertAlmostEqual(results.residuals["two"][0]['Y'][i], resid)
            self.assertAlmostEqual(results.conditioned_residuals["two"][0]['Y'][i], resid/2)

        state_one_resids = []
        for i in range(len(results.weighted_conditioned_residuals["one"][0]["Y"])):
            resid = i/4*(2.0-3.0)/3
            state_one_resids.append(resid)
            self.assertAlmostEqual(results.weighted_conditioned_residuals["one"][0]["Y"][i], resid)
        
        state_two_resids = []
        for i in range(len(results.weighted_conditioned_residuals["two"][0]["Y"])):
            resid = i/4*(3.0-2.0)/2
            state_two_resids.append(resid)
            self.assertAlmostEqual(results.weighted_conditioned_residuals["two"][0]['Y'][i], resid)
        
        state_one_resids_norm = state_one_resids /  np.sqrt(len(state_one_resids))
        state_one_obj = np.linalg.norm(state_one_resids_norm)**2
        self.assertAlmostEqual(state_one_obj, results.objectives["one"][0]["Y"][0])

        state_two_resids_norm = state_two_resids /  np.sqrt(len(state_two_resids))
        state_two_obj = np.linalg.norm(state_two_resids_norm)**2
        self.assertAlmostEqual(state_two_obj, results.objectives["two"][0]["Y"][0])

        total_residual_goal = np.concatenate([state_two_resids, state_one_resids])
        total_residual_goal /= np.sqrt(len(total_residual_goal))
        self.assert_close_arrays(total_residual_goal, results.calibration_residuals)

        total_objective_goal = np.linalg.norm(total_residual_goal)**2
        self.assertAlmostEqual(total_objective_goal, results.get_objective())
        
        for i in range(len(qois.simulation_qois["one"][0]["Y"])):
            x = i/4.0
            y =2*x
            self.assertAlmostEqual(qois.simulation_qois["one"][0]["Y"][i], y)
            self.assertAlmostEqual(qois.conditioned_simulation_qois["one"][0]["Y"][i], y/3.0)
            self.assertAlmostEqual(qois.weighted_conditioned_simulation_qois["one"][0]["Y"][i], y/3.0)
            
        for i in range(len(qois.simulation_qois["two"][0]["Y"])):
            x = i/4
            y =3*x
            self.assertAlmostEqual(qois.simulation_qois["two"][0]["Y"][i], y)
            self.assertAlmostEqual(qois.conditioned_simulation_qois["two"][0]["Y"][i], y/2.0)
            self.assertAlmostEqual(qois.weighted_conditioned_simulation_qois["two"][0]["Y"][i], y/2.0)

        for i in range(len(qois.experiment_qois["one"][0]["Y"])):
            x = i/4.0
            y =3*x
            self.assertAlmostEqual(qois.experiment_qois["one"][0]["Y"][i], y)
            self.assertAlmostEqual(qois.conditioned_experiment_qois["one"][0]["Y"][i], y/3.0)
            self.assertAlmostEqual(qois.weighted_conditioned_experiment_qois["one"][0]["Y"][i], y/3.0)
            
        for i in range(len(qois.experiment_qois["two"][0]["Y"])):
            x = i/4
            y =2*x
            self.assertAlmostEqual(qois.experiment_qois["two"][0]["Y"][i], y)
            self.assertAlmostEqual(qois.conditioned_experiment_qois["two"][0]["Y"][i], y/2.0)
            self.assertAlmostEqual(qois.weighted_conditioned_experiment_qois["two"][0]["Y"][i], y/2.0)

        
        self.assertEqual(qois.experiment_data.field_names, ["X", "Y", "Z"])
        self.assertEqual(qois.simulation_data.field_names, ["X", "Y", "Z"])

        self.assertEqual(qois.experiment_qois.field_names, ["X", "Y", "Z"])
        self.assertEqual(qois.conditioned_experiment_qois.field_names, ["X", "Y", "Z"])
        self.assertEqual(qois.weighted_conditioned_experiment_qois.field_names, ["X", "Y", "Z"])

        self.assertEqual(qois.simulation_qois.field_names, ["X", "Y"])
        self.assertEqual(qois.conditioned_simulation_qois.field_names, ["X", "Y"])
        self.assertEqual(qois.weighted_conditioned_simulation_qois.field_names, ["X", "Y"])

        self.assertEqual(results.residuals.field_names, ["Y"])
        self.assertEqual(results.conditioned_residuals.field_names, ["Y"])
        self.assertEqual(results.weighted_conditioned_residuals.field_names, ["Y"])

        self.assertEqual(results.objectives.field_names, ["Y"])

    def _make_extensive_results(self):
        cbol2 = CurveBasedInterpolatedObjective("X", "Y")

        state1 = State("one")
        state2 = State("two")

        data_vals = np.linspace(0, 1, 5)
        data_2_vals = np.linspace(0, 2, 5)
        data_dict = {"X":data_vals, "Y":data_2_vals, "Z":data_vals}
        data = convert_dictionary_to_data(data_dict)
        data.set_state(state2)
                
        data_dict_2 = {"X":data_vals, "Y":1.5*data_2_vals, "Z":data_vals}
        data_2 = convert_dictionary_to_data(data_dict_2)
        data_2.set_state(state1)

        data_long_vals = np.linspace(0, 1, 50)
        data_long_2_vals = np.linspace(0, 2, 50)
        data_dict_long = {"X":data_long_vals, "Y":data_long_2_vals, "Z":data_long_vals}
        data_long = convert_dictionary_to_data(data_dict_long)
        data_long.set_state(state1)
        
        data_dict_long_2 = {"X":data_long_vals, "Y":1.5*data_long_2_vals, "Z":data_long_vals}
        data_long_2 = convert_dictionary_to_data(data_dict_long_2)
        data_long_2.set_state(state2)

        sim_qois =  data.copy()
        sim_qois.set_state(state1)
        sim_qois = sim_qois.remove_field("Z")
        sim_qois_2 = data_2.copy()
        sim_qois_2.set_state(state2)
        sim_qois_2 = sim_qois_2.remove_field("Z")

        short_dc = DataCollection("ref", data, data_2)
        long_dc = DataCollection("work", data_long, data_long_2)
        sim_qois_dc = DataCollection("qois work", sim_qois_2, sim_qois)

        data_conditioner = RangeDataConditioner()
        data_conditioner.initialize_data_conditioning_values([data_2])
        cond_data_2 = data_conditioner.apply_to_data(data_2)
        cond_data_long = data_conditioner.apply_to_data(data_long)
        cond_sim_qois = data_conditioner.apply_to_data(sim_qois)

        data_conditioner.initialize_data_conditioning_values([data])
        cond_data = data_conditioner.apply_to_data(data)
        cond_data_long_2 = data_conditioner.apply_to_data(data_long_2)
        cond_sim_qois_2 = data_conditioner.apply_to_data(sim_qois_2)

        cond_short_dc = DataCollection("conditioned ref", cond_data, cond_data_2)
        cond_long_dc = DataCollection("conditioned work", cond_data_long, cond_data_long_2)
        cond_sim_qois_dc = DataCollection("conditioned qois work", cond_sim_qois_2, cond_sim_qois)

        obj_set = ObjectiveSet(ObjectiveCollection("test", cbol2), short_dc, short_dc.states)
        set_results = obj_set.calculate_objective_set_results(long_dc)
        results = set_results[0][cbol2.name]
        qois = set_results[1][cbol2.name]
        return short_dc, long_dc, sim_qois_dc, cond_short_dc, cond_long_dc, cond_sim_qois_dc, results, qois
        
    def _assert_dictionary_values_are_the_same(self, r_qoi, dump_qoi, dofs):
        for state, state_data in r_qoi.items():
            for r_data, d_data in zip(state_data, dump_qoi[state.name]):
                for dof in dofs:
                    self.assert_close_arrays(r_data[dof], d_data[dof])
