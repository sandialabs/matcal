import numpy as np
from copy import copy, deepcopy
                                  
from matcal.core.data import (Data, DataCollection, ReturnPassedDataConditioner, 
                             convert_dictionary_to_data)
from matcal.core.objective import (CurveBasedInterpolatedObjective,
                                  DirectCurveBasedInterpolatedObjective, Objective, 
                                  ObjectiveCollection, ObjectiveSet,
                                  L1NormMetricFunction, L2NormMetricFunction, 
                                  NormMetricFunction, SumSquaresMetricFunction, 
                                  SimulationResultsSynchronizer)
from matcal.core.qoi_extractor import MaxExtractor
from matcal.core.residuals import (ConstantFactorWeighting, IdentityWeighting, 
                                   LogResidualCalculator, UserFunctionWeighting, 
                                   NominalResidualCalculator, get_array)
from matcal.core.state import SolitaryState, State, StateCollection
from matcal.core.tests.MatcalUnitTest import MatcalUnitTest


class MetricFunctionTest(MatcalUnitTest):

    def setUp(self) -> None:
        super().setUp(__file__)

        self.state = State("ex")
        data_dict = {"x":np.linspace(0.25, 10, 40), "y":np.linspace(0.25, 10, 40)}
        self.sim_data_frame = convert_dictionary_to_data(data_dict)
        self.sim_dataset = Data(self.sim_data_frame)
        self.sim_dataset.set_state(self.state)

        exp_dict = {"x":np.linspace(1, 10, 40), "y":2 * np.linspace(1, 10, 40)}
        self.exp_data_frame = convert_dictionary_to_data(exp_dict)
        self.exp_dataset = Data(self.exp_data_frame)
        self._res = NominalResidualCalculator("y")

    def test_l2norm_norm_and_calculate(self):
        metric = L2NormMetricFunction()
        obj_function_result = metric(get_array(self._res.calculate(self.exp_dataset, self.sim_dataset)))
        true_result =  np.linalg.norm((self.sim_data_frame["y"] - self.exp_data_frame["y"]) / np.sqrt(len(self.exp_data_frame['y'])))

        self.assertAlmostEqual(obj_function_result, true_result, delta=1e-6)

    def test_l2norm_normalize(self):
        metric = L2NormMetricFunction()
        n_res =  metric.normalize_only(get_array(self._res.calculate(self.exp_dataset, self.sim_dataset)))
        true_result =  (self.sim_data_frame["y"] - self.exp_data_frame["y"]) / np.sqrt(len(self.exp_data_frame['y']))
        self.assert_close_arrays(n_res, true_result)
    
    def test_l2norm_calculate(self):
        metric = L2NormMetricFunction()
        n_res =  metric.calculate_only(get_array(self._res.calculate(self.exp_dataset, self.sim_dataset)))
        true_result =  np.linalg.norm(self.sim_data_frame["y"] - self.exp_data_frame["y"])
        self.assertAlmostEqual(n_res, true_result)

    def test_l1norm_normalize(self):
        metric = L1NormMetricFunction()
        n_res =  metric.normalize_only(get_array(self._res.calculate(self.exp_dataset, self.sim_dataset)))
        true_result =  (self.sim_data_frame["y"] - self.exp_data_frame["y"]) / len(self.exp_data_frame['y'])
        self.assert_close_arrays(n_res, true_result)

    def test_l1norm_normalize_and_calculate(self):
        func = L1NormMetricFunction()
        resids = self._res.calculate(self.exp_dataset, self.sim_dataset)
        obj_function_result = func(get_array(resids))
        true_result =  np.linalg.norm((self.sim_data_frame["y"] - self.exp_data_frame["y"]) / len(self.exp_data_frame['y']), 1)

        self.assertAlmostEqual(obj_function_result, true_result, delta=1e-6)

    def test_l1norm_calculate(self):
        func = L1NormMetricFunction()
        resids = self._res.calculate(self.exp_dataset, self.sim_dataset)
        obj_function_result = func.calculate_only(get_array(resids))
        true_result =  np.linalg.norm((self.sim_data_frame["y"] - self.exp_data_frame["y"]), 1)

        self.assertAlmostEqual(obj_function_result, true_result, delta=1e-6)

    def test_general_norm_metric_function_bad_init(self):
        with self.assertRaises(NormMetricFunction.OrderError):
            func = NormMetricFunction('abc')

    def test_sum_of_squares_metric_function_calculate(self):
        metric = SumSquaresMetricFunction()
        resids = self._res.calculate(self.exp_dataset, self.sim_dataset)
        obj_function_result = metric.calculate_only(get_array(resids))
        true_result =  np.linalg.norm((self.sim_data_frame["y"] - self.exp_data_frame["y"]))**2
        self.assertAlmostEqual(obj_function_result, true_result, delta=1e-6)

    def test_sum_of_squares_metric_function_calculate_and_normalize(self):
        metric = SumSquaresMetricFunction()
        resids = self._res.calculate(self.exp_dataset, self.sim_dataset)
        obj_function_result = metric.normalize_and_calculate(get_array(resids))
        true_result =  np.linalg.norm((self.sim_data_frame["y"] - self.exp_data_frame["y"])
                                      /np.sqrt(len(self.exp_data_frame["y"])))**2
        self.assertAlmostEqual(obj_function_result, true_result, delta=1e-6)

    def test_sum_of_squares_metric_function_normalize(self):
        metric = SumSquaresMetricFunction()
        resids = self._res.calculate(self.exp_dataset, self.sim_dataset)
        obj_function_result = metric.normalize_only(get_array(resids))
        self.assert_close_arrays(obj_function_result, get_array(resids)/np.sqrt(len(resids)))


def user_weight_func(indep_var, dep_avr, resids):
    import numpy as np
    weights = indep_var*0+1
    return resids*weights


class TestObjective(MatcalUnitTest):
    def setUp(self):
        super().setUp(__file__)
        self.state = SolitaryState()

        data = np.linspace(0, 1, 5)
        data_2 = np.linspace(0, 2, 5)
        data_dict = {"X":data, "Y":data_2, "Z":data}
        self.data = convert_dictionary_to_data(data_dict)
        self.data.set_state(self.state)
                
        data_dict_2 = {"X":1.5*data, "Y":1.5*data_2, "Z":1.5*data}
        self.data_2 = convert_dictionary_to_data(data_dict_2)
        self.data_2.set_state(self.state)

        self.data_mat = np.column_stack((data, data_2, data))
        self.data_mat_3 = self.data_mat * self.data_mat
        data_dict_3 = {"X":self.data_mat_3[:,0], "Y":self.data_mat_3[:,1], "Z":self.data_mat_3[:,2]}
        self.data_3 = convert_dictionary_to_data(data_dict_3)
        self.data_3.set_state(self.state)
      
        single_row_ones_dict = {"X":[1], "Y":[1]}
        self.data_single_row_ones = convert_dictionary_to_data(single_row_ones_dict)
        self.data_single_row_ones.set_state(self.state)

        single_row_twos_dict = {"Z":[2], "Y":[2], "X":[2]}
        self.data_single_row_twos = convert_dictionary_to_data(single_row_twos_dict)
        self.data_single_row_twos.set_state(self.state)

    def test_init(self):
        ob = Objective("x")

    def test_bad_init(self):
        with self.assertRaises(Objective.TypeError):
            ob = Objective(1)
            ob = Objective(None)

    def test_has_independent_field(self):
        ob = Objective("x")
        self.assertFalse(ob.has_independent_field())

    def test_add_qoi_extractors(self):
        ob = Objective("X", "Y")
        extractor = MaxExtractor("Y")
        ob.set_qoi_extractors(extractor)
        with self.assertRaises(Objective.TypeError):
            ob.set_qoi_extractors(None)
            ob.set_qoi_extractors(1)
    
    def test_vector_minus_scalar(self):
        ob = Objective("X", "Y")
        vector_data_dict = {"X":np.linspace(0,1,10), "Y":2*np.linspace(0, 10, 10)}
        scalar_data_dict = {"X":1, "Y":0}

        vector_data = convert_dictionary_to_data(vector_data_dict)
        scalar_data = convert_dictionary_to_data(scalar_data_dict)

        vector_data_collection = DataCollection("vector", vector_data)
        scalar_data_collection = DataCollection("scalar", scalar_data)

        with self.assertRaises(Objective.InconsistentDataError):
            results, qois = ob.calculate(vector_data_collection, vector_data_collection, scalar_data_collection, scalar_data_collection)

    def test_no_qoi_extractor(self):
        ob = Objective("X", "Y")
        exp_collection = DataCollection("exp", self.data)
        sim_collection = DataCollection("sim", self.data_3)

        ob_set = ObjectiveSet(ObjectiveCollection("test", ob), exp_collection, exp_collection.states)
        results, qois = ob_set.calculate_objective_set_results(sim_collection)
        results = results[ob.name]        

        target_x = self.data_mat_3[:, 0] - self.data_mat[:, 0]
        target_y = self.data_mat_3[:, 1] - self.data_mat[:, 1]

        for i, res in enumerate(results.residuals[self.state.name]):
            for j, res_val in enumerate(res["X"]):
                self.assertAlmostEqual(res_val, target_x[j])

        for i, res in enumerate(results.residuals[self.state.name]):
            for j, res_val in enumerate(res["Y"]):
                self.assertAlmostEqual(res_val, target_y[j])

    def test_user_weights(self):
        user_weight = UserFunctionWeighting("X", "Y", user_weight_func)
        ob = Objective("X", "Y")
        ob.set_field_weights(user_weight)
        exp_collection = DataCollection("exp", self.data)
        sim_collection = DataCollection("sim", self.data_3)

        ob_set = ObjectiveSet(ObjectiveCollection("test", ob), exp_collection, exp_collection.states)
        results, qois = ob_set.calculate_objective_set_results(sim_collection)
        results = results[ob.name]

        target_x = []
        target_y = []

        target_x.append((self.data_mat_3[:, 0] - self.data_mat[:, 0]))
        target_y.append((self.data_mat_3[:, 1] - self.data_mat[:, 1]))

        for i, res in enumerate(results.weighted_residuals[self.state.name]):
            for j, res_val in enumerate(res["X"]):
                self.assertAlmostEqual(res_val, target_x[i][j])

        for i, res in enumerate(results.weighted_residuals[self.state.name]):
            for j, res_val in enumerate(res["Y"]):
                self.assertAlmostEqual(res_val, target_y[i][j])

    def test_no_qoi_extractor_mult_data(self):
        ob = Objective("X", "Y")
        exp_collection = DataCollection("exp", self.data, self.data_2)
        sim_collection = DataCollection("sim", self.data_3)

        ob_set = ObjectiveSet(ObjectiveCollection("test", ob), exp_collection, exp_collection.states)
        results, qois = ob_set.calculate_objective_set_results(sim_collection)
        results = results[ob.name]

        target_x = [self.data_mat_3[:, 0] - self.data_mat[:, 0]] 
        target_x.append(self.data_mat_3[:, 0] - self.data_2["X"])
        target_y = [self.data_mat_3[:, 1] - self.data_mat[:, 1]]
        target_y.append(self.data_mat_3[:, 1] - self.data_2["Y"])

        for i, res in enumerate(results.residuals[self.state]):
            for j, res_val in enumerate(res["Y"]):
                self.assertAlmostEqual(res_val, target_y[i][j])

        for i, res in enumerate(results.residuals[self.state]):
            for j, res_val in enumerate(res["X"]):
                self.assertAlmostEqual(res_val, target_x[i][j])

    def test_max_qoi_extractors_mult_data(self):
        ob = Objective("X", "Y")
        ob.set_qoi_extractors(MaxExtractor("X"))
        exp_collection = DataCollection("exp", self.data, self.data_2)
        sim_collection = DataCollection("sim", self.data_3)

        ob_set = ObjectiveSet(ObjectiveCollection("test", ob), exp_collection, exp_collection.states)
        results, qois = ob_set.calculate_objective_set_results(sim_collection)
        results = results[ob.name]

        target_x = [self.data_mat_3[-1, 0] - self.data_mat[-1, 0]]
        target_y = [self.data_mat_3[-1, 1] - self.data_mat[-1, 1]]
        target_x.append(self.data_mat_3[-1, 0] - self.data_2["X"][-1])
        target_y.append(self.data_mat_3[-1, 1] - self.data_2["Y"][-1])
        
        for i, res in enumerate(results.residuals[self.state.name]):
            for j, res_val in enumerate(res["X"]):
                self.assertAlmostEqual(res_val, target_x[i])
        for i, res in enumerate(results.residuals[self.state.name]):
            for j, res_val in enumerate(res["Y"]):
                self.assertAlmostEqual(res_val, target_y[i])

    def test_max_exp_qoi_extractor_mult_data(self):
        ob = Objective("X", "Y")
        ob.set_experiment_qoi_extractor(MaxExtractor("X"))
        exp_collection = DataCollection("exp", self.data, self.data_2)
        sim_collection = DataCollection("sim", self.data_single_row_ones)

        ob_set = ObjectiveSet(ObjectiveCollection("test", ob), exp_collection, exp_collection.states)
        results, qois = ob_set.calculate_objective_set_results(sim_collection)
        results = results[ob.name]

        target_x = [1 - self.data_mat[-1, 0]]
        target_y = [ 1 - self.data_mat[-1, 1]]
        target_x.append(1 - self.data_2["X"][-1])
        target_y.append(1 - self.data_2["Y"][-1])

        for i, res in enumerate(results.residuals[self.state.name]):
            for j, res_val in enumerate(res["X"]):
                self.assertAlmostEqual(res_val, target_x[i])
        for i, res in enumerate(results.residuals[self.state.name]):
            for j, res_val in enumerate(res["Y"]):
                self.assertAlmostEqual(res_val, target_y[i])

    def test_max_sim_qoi_extractor_mult_data(self):
        ob = Objective("X", "Y")
        ob.set_simulation_qoi_extractor(MaxExtractor("X"))
        exp_collection = DataCollection("exp", self.data_single_row_ones, self.data_single_row_twos)
        sim_collection = DataCollection("sim", self.data_3)

        ob_set = ObjectiveSet(ObjectiveCollection("test", ob), exp_collection, exp_collection.states)
        results, qois = ob_set.calculate_objective_set_results(sim_collection)
        results = results[ob.name]

        target_x = [self.data_mat_3[-1, 0] - 1]
        target_y = [self.data_mat_3[-1, 1] - 1]
        target_x.append(self.data_mat_3[-1, 0] - 2)
        target_y.append(self.data_mat_3[-1, 1] - 2)

        for i, res in enumerate(results.residuals[self.state.name]):
            for j, res_val in enumerate(res["X"]):
                self.assertAlmostEqual(res_val, target_x[i])
        for i, res in enumerate(results.residuals[self.state.name]):
            for j, res_val in enumerate(res["Y"]):
                self.assertAlmostEqual(res_val, target_y[i])

    def test_missing_required_field_sim(self):
        ob = Objective("X", "Y", "Z")
        exp_collection = DataCollection("exp", self.data)
        sim_collection = DataCollection("sim", self.data_single_row_ones)

        with self.assertRaises(Objective.MissingFieldsError):
            results, qois = ob.calculate(exp_collection, exp_collection, sim_collection, sim_collection)

    def test_missing_required_field_exp(self):
        ob = Objective("X", "Y", "Z")
        sim_collection = DataCollection("sim", self.data)
        exp_collection = DataCollection("exp", self.data_single_row_ones)

        with self.assertRaises(Objective.MissingFieldsError):
            results, qois = ob.calculate(exp_collection, exp_collection, sim_collection, sim_collection)

    def test_name(self):
        ob = Objective("X", "Y", "Z")
        self.assertEqual(ob.name.split('_')[0], "Objective" )
        
        ob.set_name("user name")
        self.assertEqual(ob.name, "user name")

    def test_has_fields_of_interest(self):
        ob = Objective("X")
        self.assertTrue(ob.has_fields_of_interest())

class TestCurveBasedObjective(MatcalUnitTest):

    def setUp(self):
        super().setUp(__file__)
        self.zeros_dict = {"X":[0,1,2,3],"Y":[0,0,0,0]}
        self.data_zeros = convert_dictionary_to_data(self.zeros_dict)
        self.state = SolitaryState()
        self.data_zeros.set_state(self.state)
        self.zero_data_collection = DataCollection("zero", self.data_zeros)

        self.ones_dict = {"X":[0,1,2,3],"Y":[1,1,1,1]}
        self.data_ones = convert_dictionary_to_data(self.ones_dict)
        self.data_ones.set_state(self.state)
        self.ones_data_collection = DataCollection("one", self.data_ones)

        single_row_ones_dict = {"Z":[1], "Y":[1]}
        self.data_single_row_ones_without_X = convert_dictionary_to_data(single_row_ones_dict)

        data = np.linspace(0, 1.0, 5)
        data_2 = np.linspace(0, 2, 5)
        data_dict = {"X":data, "Y":data_2, "Z":data}

        self.data = convert_dictionary_to_data(data_dict)
        self.data.set_state(self.state)
        
        data_dict_2 = {"X":1.5*data, "Y":1.5*data_2, "Z":1.5*data}
        self.data_2 = convert_dictionary_to_data(data_dict_2)
        self.data_2.set_state(self.state)

        self.data_mat = np.vstack((data, data_2, data)).T
        self.data_mat_3 = self.data_mat * self.data_mat
        data_dict_3 = {"X":self.data_mat_3[:,0], "Y":self.data_mat_3[:,1], "Z":self.data_mat_3[:,2]}
        self.data_3 = convert_dictionary_to_data(data_dict_3)
        self.data_3.set_state(self.state)

        data_dict_double_row_ones = {"X":[1, 1], "Y":[1,1]}
        self.data_double_row_ones = convert_dictionary_to_data(data_dict_double_row_ones)
        self.data_double_row_ones.set_state(self.state)

    def test_init_with_left_right_period(self):
        with self.assertRaises(CurveBasedInterpolatedObjective.TypeError):
            obj = CurveBasedInterpolatedObjective("X", "Y", left="a")

    def test_default_residual_calculator_is_nominal(self):
        cbol2 = CurveBasedInterpolatedObjective("X", "Y")
        self.assertIsInstance(cbol2._residual_calculator, NominalResidualCalculator)

    def test_convert_to_log_residual(self):
        cbol2 = CurveBasedInterpolatedObjective("X", "Y")
        cbol2.use_log_residual()
        self.assertIsInstance(cbol2._residual_calculator, LogResidualCalculator)

    def test_name(self):
        cbol2 = CurveBasedInterpolatedObjective("X", "Y")
        self.assertTrue("CurveBasedInterpolatedObjective" in cbol2.name.split("_"))

    def test_has_independent_field(self):
        cbol2 = CurveBasedInterpolatedObjective("X", "Y")
        self.assertTrue(cbol2.has_independent_field())

    def test_zero_calculate(self):
        cbol2 = CurveBasedInterpolatedObjective("X", "Y")
        results, qois = cbol2.calculate(self.zero_data_collection, self.zero_data_collection, self.zero_data_collection,
                                  self.zero_data_collection)

        for i in range(len(results.weighted_conditioned_residuals[self.state][0]["Y"])):
            self.assertAlmostEqual(results.weighted_conditioned_residuals[self.state.name][0]["Y"][i], 0.0)

    def test_ones_calculate(self):
        cbol2 = CurveBasedInterpolatedObjective("X", "Y")

        results, qois = cbol2.calculate(self.zero_data_collection, 
                                        self.zero_data_collection, self.ones_data_collection,
                                        self.ones_data_collection)

        for i in range(len(results.weighted_conditioned_residuals[self.state][0]["Y"])):
            self.assertAlmostEqual(results.weighted_conditioned_residuals[self.state.name][0]["Y"][i], 1)
        n_pts = 4
        goal_full = np.ones(n_pts) / np.sqrt(n_pts)
        self.assert_close_arrays(goal_full, results.calibration_residuals, show_on_fail=True)
        self.assertAlmostEqual(1, results.get_objective())

    def test_user_weights(self):
        user_weight = UserFunctionWeighting("X", "Y", user_weight_func)
        ob = CurveBasedInterpolatedObjective("X", "Y")
        ob.set_field_weights( user_weight)
        exp_collection = DataCollection("exp", self.data, self.data_double_row_ones)
        sim_collection = DataCollection("sim", self.data_3)

        ob_set = ObjectiveSet(ObjectiveCollection("test", ob), exp_collection, exp_collection.states, ReturnPassedDataConditioner)
        results, qois = ob_set.calculate_objective_set_results(sim_collection)
        results = results[ob.name]

        target_y = []
        target_y.append((np.interp(self.data_mat[:, 0],  self.data_mat_3[:, 0], self.data_mat_3[:, 1]) - self.data_mat[:, 1]))
        target_y.append((np.interp(self.data_double_row_ones["X"], self.data_mat_3[:, 0], self.data_mat_3[:, 1]) - self.data_double_row_ones["Y"]))

        for i, res in enumerate(results.weighted_conditioned_residuals[self.state.name]):
            for j, res_val in enumerate(res["Y"]):
                self.assertAlmostEqual(res_val, target_y[i][j])

    def test_multi_states(self):
        cbol2 = CurveBasedInterpolatedObjective("X", "Y")
        s1 = State("one")
        s2 = State("two")
        d1 = convert_dictionary_to_data(self.zeros_dict)
        d1.set_state(s1)
        d2 = convert_dictionary_to_data(self.zeros_dict)
        d2.set_state(s2)
        d3 = convert_dictionary_to_data(self.zeros_dict)
        d3.set_state(s1)
        d4 = convert_dictionary_to_data(self.ones_dict)
        d4.set_state(s2)

        ref_dc = DataCollection("ref", d1, d2)
        work_dc = DataCollection("work", d3, d4)
        results, qois = cbol2.calculate(ref_dc, ref_dc, work_dc, work_dc)

        for i in range(len(results.weighted_conditioned_residuals["two"][0]["Y"])):
            self.assertAlmostEqual(results.weighted_conditioned_residuals["two"][0]["Y"][i], 1)
        for i in range(len(results.weighted_conditioned_residuals["one"][0]["Y"])):
            self.assertAlmostEqual(results.weighted_conditioned_residuals["one"][0]['Y'][i], 0)
        
        data_points_per_set = 4
        full_residual = np.ones(data_points_per_set * 2)
        full_residual[:data_points_per_set] = 0
        full_residual /= np.sqrt(data_points_per_set*2)
        self.assert_close_arrays(results.calibration_residuals, full_residual)

    def test_interpolate_simulation(self):
        cbol2 = CurveBasedInterpolatedObjective("X", "Y")

        state1 = State("one")
        state2 = State("two")

        data_vec = np.linspace(0, 1, 5)
        data_vec_2 = np.linspace(0, 2, 5)
        
        data_dict = {"X":data_vec, "Y":data_vec_2, "Z":data_vec}
        data = convert_dictionary_to_data(data_dict)
        data.set_state(state2)

        data_2 = data.copy()
        data_2["Y"] *= 1.5
        data_2.set_state(state1)

        data_long_vec = np.linspace(0, 1, 50)
        data_long_vec_2 = np.linspace(0, 2, 50)
        data_long_dict = {"X":data_long_vec, "Y":data_long_vec_2, "Z":data_long_vec}
        data_long = convert_dictionary_to_data(data_long_dict)
        data_long.set_state(state1)
        
        data_long_2 = data_long.copy()
        data_long_2["Y"] *= 1.5
        data_long_2.set_state(state2)
        
        short_dc = DataCollection("ref", data, data_2)
        long_dc = DataCollection("work", data_long, data_long_2)

        ob_set = ObjectiveSet(ObjectiveCollection("test", cbol2), short_dc, short_dc.states, ReturnPassedDataConditioner)
        oset_results, oset_qois = ob_set.calculate_objective_set_results(long_dc)
        results = oset_results[cbol2.name]
        qois = oset_qois[cbol2.name]

        self.assertEqual(len(qois.get_flattened_weighted_conditioned_experiment_qois()), 10)
        self.assertEqual(len(qois.get_flattened_weighted_conditioned_simulation_qois()), 10)
        for i in range(len(results.weighted_conditioned_residuals["one"][0]["Y"])):
            self.assertAlmostEqual(results.weighted_conditioned_residuals["one"][0]["Y"][i], i/4*(2.0-3.0))
        
        for i in range(len(results.weighted_conditioned_residuals["two"][0]["Y"])):
            self.assertAlmostEqual(results.weighted_conditioned_residuals["two"][0]['Y'][i], i/4*(3.0-2.0))

    def test_extrapolate_simulation_with_interp_right_zero(self):
        cbol2 = CurveBasedInterpolatedObjective("X", "Y", right=10)

        state1 = State("one")
        state2 = State("two")

        data_vec = np.linspace(0, 2, 9)
        data_vec_2 = np.linspace(0, 4, 9)

        data_dict = {"X":data_vec, "Y":data_vec_2, "Z":data_vec}
        data = convert_dictionary_to_data(data_dict)
        data.set_state(state2)

        data_2 = data.copy()
        data_2["Y"] *= 1.5
        data_2.set_state(state1)

        data_long_vec = np.linspace(0, 1, 50)
        data_long_vec_2 = np.linspace(0, 2, 50)
        data_long_dict = {"X":data_long_vec, "Y":data_long_vec_2, "Z":data_long_vec}
        data_long = convert_dictionary_to_data(data_long_dict)
        data_long.set_state(state1)
        
        data_long_2 = data_long.copy()
        data_long_2["Y"] *= 1.5
        data_long_2.set_state(state2)
        
        short_dc = DataCollection("ref", data, data_2)
        long_dc = DataCollection("work", data_long, data_long_2)

        ob_set = ObjectiveSet(ObjectiveCollection("test", cbol2), short_dc, short_dc.states, ReturnPassedDataConditioner)
        oset_results, oset_qois = ob_set.calculate_objective_set_results(long_dc)
        results = oset_results[cbol2.name]
        qois = oset_qois[cbol2.name]

        self.assertEqual(len(qois.get_flattened_weighted_conditioned_experiment_qois()), 18)
        self.assertEqual(len(qois.get_flattened_weighted_conditioned_simulation_qois()), 18)
        for i in range(5):
            self.assertAlmostEqual(results.weighted_conditioned_residuals["one"][0]["Y"][i], i/4*(2.0-3.0))
        for i in range(5,9):
            self.assertAlmostEqual(results.weighted_conditioned_residuals["one"][0]["Y"][i], 10-(i/4*3.0))

        for i in range(5):
            self.assertAlmostEqual(results.weighted_conditioned_residuals["two"][0]['Y'][i], i/4*(3.0-2.0))
        for i in range(5,9):
            self.assertAlmostEqual(results.weighted_conditioned_residuals["two"][0]["Y"][i], 10-(i/4*2.0))
            
    def test_extrapolate_simulation_with_interp_right_none(self):
        cbol2 = CurveBasedInterpolatedObjective("X", "Y", right=None)

        state1 = State("one")
        state2 = State("two")

        data_vec = np.linspace(0, 2, 9)
        data_vec_2 = np.linspace(0, 4, 9)

        data_dict = {"X":data_vec, "Y":data_vec_2, "Z":data_vec}
        data = convert_dictionary_to_data(data_dict)
        data.set_state(state2)

        data_2 = data.copy()
        data_2["Y"] *= 1.5
        data_2.set_state(state1)

        data_long_vec = np.linspace(0, 1, 50)
        data_long_vec_2 = np.linspace(0, 2, 50)
        data_long_dict = {"X":data_long_vec, "Y":data_long_vec_2, "Z":data_long_vec}
        data_long = convert_dictionary_to_data(data_long_dict)
        data_long.set_state(state1)
        
        data_long_2 = data_long.copy()
        data_long_2["Y"] *= 1.5
        data_long_2.set_state(state2)
        
        short_dc = DataCollection("ref", data, data_2)
        long_dc = DataCollection("work", data_long, data_long_2)

        ob_set = ObjectiveSet(ObjectiveCollection("test", cbol2), short_dc, short_dc.states, ReturnPassedDataConditioner)
        oset_results, oset_qois = ob_set.calculate_objective_set_results(long_dc)
        results = oset_results[cbol2.name]
        qois = oset_qois[cbol2.name]

        self.assertEqual(len(qois.get_flattened_weighted_conditioned_experiment_qois()), 18)
        self.assertEqual(len(qois.get_flattened_weighted_conditioned_simulation_qois()), 18)
        for i in range(5):
            self.assertAlmostEqual(results.weighted_conditioned_residuals["one"][0]["Y"][i], i/4*(2.0-3.0))
        for i in range(5,9):
            self.assertAlmostEqual(results.weighted_conditioned_residuals["one"][0]["Y"][i], 2-(i/4*3.0))
        for i in range(5):
            self.assertAlmostEqual(results.weighted_conditioned_residuals["two"][0]['Y'][i], i/4*(3.0-2.0))
        for i in range(5,9):
            self.assertAlmostEqual(results.weighted_conditioned_residuals["two"][0]["Y"][i], 3-(i/4*2.0))

    def test_multiple_ref(self):
        cbol2 = CurveBasedInterpolatedObjective("X", "Y")

        s1 = State("one")
        d1 = convert_dictionary_to_data(self.zeros_dict)
        d1.set_state(s1)
        d2 = convert_dictionary_to_data(self.ones_dict)
        d2.set_state(s1)
        d3 = convert_dictionary_to_data(self.zeros_dict)
        d3.set_state(s1)

        exp_dc = DataCollection("ref", d1, d2)
        sim_dc = DataCollection("work", d3)
       
        obj_set = ObjectiveSet(ObjectiveCollection("test", cbol2), exp_dc, exp_dc.states, ReturnPassedDataConditioner)
        obj_results, obj_qois = obj_set.calculate_objective_set_results(sim_dc)
        results = obj_results[cbol2.name]

        cnt = 0
        for j in range(len(exp_dc[s1])):
            for i in range(4):
                a = 0
                if j > 0:
                    a = -1 
                self.assertAlmostEqual(results.weighted_conditioned_residuals[s1][j]["Y"][i], a)

    def test_count_entry_weights(self):
        cbol = CurveBasedInterpolatedObjective("X", "Y")
        cbol.set_field_weights(*[])
        work_data = convert_dictionary_to_data(self.zeros_dict)
        Xs = np.linspace(-5,5,11)
        Ys = 2*Xs
        exp_data_dict ={"X":Xs, "Y":Ys}
        exp_data = convert_dictionary_to_data(exp_data_dict)
        exp_dc = DataCollection("exp", exp_data)
        sim_dc = DataCollection("work", work_data)
        obj_set = ObjectiveSet(ObjectiveCollection("test", cbol), exp_dc, exp_dc.states, ReturnPassedDataConditioner)
        oset_results, oset_qois = obj_set.calculate_objective_set_results(sim_dc)
        results = oset_results[cbol.name]
        qois = oset_qois[cbol.name]
        goal = -np.linspace(-10, 10, 11)
        self.assertTrue(np.allclose(results.get_flattened_weighted_conditioned_residuals(), goal))

    def test_multiple_weights(self):
        cbol = CurveBasedInterpolatedObjective("X", "Y")
        cbol.set_field_weights(ConstantFactorWeighting(1.0 / 11.0), ConstantFactorWeighting(1.0 / 2.0))
        work_data = convert_dictionary_to_data(self.zeros_dict)
        Xs = np.linspace(-5,5,11)
        Ys = 2*Xs
        exp_data_dict ={"X":Xs, "Y":Ys}
        exp_data = convert_dictionary_to_data(exp_data_dict)

        exp_dc = DataCollection("exp", exp_data)
        sim_dc = DataCollection("work", work_data)
        obj_set = ObjectiveSet(ObjectiveCollection("test", cbol), exp_dc, exp_dc.states, ReturnPassedDataConditioner)
        results = obj_set.calculate_objective_set_results(sim_dc)[0][cbol.name]

        goal = -np.linspace(-10, 10, 11) / 2 / 11.0
        self.assertTrue(np.allclose(results.get_flattened_weighted_conditioned_residuals(), goal))

    def test_multiple_weights_more_weights(self):

        def user_weights_funct(x, y, resids):
            import numpy as np
            weights = np.linspace(0, np.pi, len(resids))
            return weights * resids

        cbol = CurveBasedInterpolatedObjective("X", "Y")
        cbol.set_field_weights(ConstantFactorWeighting(1.0 / 22.0),
                               UserFunctionWeighting("X", "Y", user_weights_funct))

        work_data = convert_dictionary_to_data(self.zeros_dict)
        Xs = np.linspace(-5,5,11)
        Ys = 2*Xs
        exp_data_dict ={"X":Xs, "Y":Ys}
        exp_data = convert_dictionary_to_data(exp_data_dict)
        exp_dc = DataCollection("exp", exp_data)
        sim_dc = DataCollection("work", work_data)
        obj_set = ObjectiveSet(ObjectiveCollection("test", cbol), exp_dc, exp_dc.states, ReturnPassedDataConditioner)
        results = obj_set.calculate_objective_set_results(sim_dc)[0][cbol.name]
        goal = -np.linspace(-10, 10, 11) / 22.0 * np.linspace(0, np.pi, 11)
        self.assertTrue(np.allclose(results.get_flattened_weighted_conditioned_residuals(), goal))

    def test_missing_independent_field_sim(self):
        ob = CurveBasedInterpolatedObjective("X", "Y")
        sim_collection = DataCollection("sim", self.data_single_row_ones_without_X)
        with self.assertRaises(CurveBasedInterpolatedObjective.MissingFieldsError):
            results = ob.calculate(self.ones_data_collection, self.ones_data_collection, sim_collection, sim_collection)

    def test_missing_independent_field_exp(self):
        ob = CurveBasedInterpolatedObjective("X", "Y")
        exp_collection = DataCollection("sim", self.data_single_row_ones_without_X)
        with self.assertRaises(CurveBasedInterpolatedObjective.MissingFieldsError):
            results = ob.calculate(exp_collection, exp_collection, self.ones_data_collection,
                                   self.ones_data_collection)

    def test_direct_objective_raise_error_if_weights_passed(self):
        ob = DirectCurveBasedInterpolatedObjective("X", "Y")
        with self.assertRaises(RuntimeError):
            ob.set_field_weights(ConstantFactorWeighting(1.0 / 22.0))

    def test_direct_objective_okay_if_identity_weight_passed(self):
        ob = DirectCurveBasedInterpolatedObjective("X", "Y")
        ob.set_field_weights(IdentityWeighting())

    def test_get_unchanged_residual(self):
        cbol = DirectCurveBasedInterpolatedObjective("X", "Y")
        
        work_data = convert_dictionary_to_data(self.zeros_dict)
        Xs = np.linspace(-5,5,11)
        Ys = 2*Xs
        exp_data_dict ={"X":Xs, "Y":Ys}
        exp_data = convert_dictionary_to_data(exp_data_dict)
        exp_dc = DataCollection("exp", exp_data)
        sim_dc = DataCollection("work", work_data)
        obj_set = ObjectiveSet(ObjectiveCollection("test", cbol), exp_dc, exp_dc.states)
        results = obj_set.calculate_objective_set_results(sim_dc)[0][cbol.name]
        goal = -Ys
        self.assert_close_arrays(results.calibration_residuals.flatten(), goal, 
                                 show_on_fail=True)


class TestSimulationResultsSynchronizer(MatcalUnitTest):

    def setUp(self):
        super().setUp(__file__)
        self.state = SolitaryState()

    def test_bad_init(self):
        with self.assertRaises(ValueError):
            res_synch = SimulationResultsSynchronizer("X", [], "Y")
        with self.assertRaises(TypeError):
            res_synch = SimulationResultsSynchronizer("X", ["a"], "Y")
        with self.assertRaises(TypeError):
            res_synch = SimulationResultsSynchronizer("X", "a", "Y")

        res_synch = SimulationResultsSynchronizer("X", 1, "Y")
            
    def test_generate_exp_data_qois_array(self):
        res_synch = SimulationResultsSynchronizer("X", np.linspace(0,1,6), "Y")
        state = self.state
        generated_data = res_synch._generate_experimental_data_qois(state)
        self.assertEqual(generated_data.state, state)
        self.assertTrue("X" in generated_data.field_names)
        self.assertTrue("Y" in generated_data.field_names)
        self.assert_close_arrays(generated_data["X"], np.linspace(0,1,6))
        self.assert_close_arrays(generated_data["Y"], np.zeros(6))

    def test_generate_exp_data_qois_single_value(self):
        res_synch = SimulationResultsSynchronizer("X", 1, "Y")
        state = self.state
        generated_data = res_synch._generate_experimental_data_qois(state)
        self.assertEqual(generated_data.state, state)
        self.assertTrue("X" in generated_data.field_names)
        self.assertTrue("Y" in generated_data.field_names)
        self.assert_close_arrays(generated_data["X"], 1)
        self.assert_close_arrays(generated_data["Y"], 0)



class TestObjectiveSet(MatcalUnitTest):
    def setUp(self):
        super().setUp(__file__)
        self.state = SolitaryState()
        self.state_2 = State("my_state")
        self.state_collection = StateCollection("states", self.state)

        data = np.linspace(-1, 1, 5)
        data_2 = np.linspace(-1, 2, 5)
        self.data_mat = np.column_stack((data, data_2))
        data_dict = {"X":self.data_mat[:,0], "Y":self.data_mat[:,1]}
        self.data_state = convert_dictionary_to_data(data_dict)
        self.data_state.set_state(self.state)

        self.data_mat_2 = 1.5 * self.data_mat
        data_dict_2 = {"X":self.data_mat_2[:,0], "Y":self.data_mat_2[:,1]}
        self.data_2_state = convert_dictionary_to_data(data_dict_2)
        self.data_2_state.set_state(self.state)
        self.data_2_state_2 = convert_dictionary_to_data(data_dict_2)
        self.data_2_state_2.set_state(self.state_2)

        self.data_mat_3 = self.data_mat * self.data_mat
        data_dict_3 = {"X":self.data_mat_3[:,0], "Y":self.data_mat_3[:,1]}
        self.data_3_state = convert_dictionary_to_data(data_dict_3)
        self.data_3_state.set_state(self.state)

        single_row_ones_dict = {"X":[1], "Y":[1]}
        self.data_single_row_ones_state = convert_dictionary_to_data(single_row_ones_dict)
        self.data_single_row_ones_state.set_state(self.state)

        single_row_twos_dict = {"Y":[2], "X":[2]}
        self.data_single_row_twos = convert_dictionary_to_data(single_row_twos_dict)
        self.data_single_row_twos.set_state(self.state)

        self.data_collection = DataCollection("my data", self.data_state, self.data_2_state,
                                              self.data_3_state, self.data_single_row_ones_state,
                                              self.data_single_row_twos)

        self.data_collection2 = DataCollection("CondTest", self.data_state, self.data_2_state_2)

        self.objective = Objective("Y")
        self.curve_objective = CurveBasedInterpolatedObjective("X", "Y")
        self.objective_collection = ObjectiveCollection("objs", self.objective, self.curve_objective)

    def test_objective_set_names(self):
        obj_set_1 = ObjectiveSet(self.objective_collection, self.data_collection2, self.data_collection2.states)
        obj_set_2 = ObjectiveSet(self.objective_collection, self.data_collection2, self.data_collection2.states)

        self.assertNotEqual(obj_set_1.name, obj_set_2.name)
        obj_set_1_num = int(obj_set_1.name.split("_")[-1])
        obj_set_2_num = int(obj_set_2.name.split("_")[-1])

        self.assertEqual(obj_set_1_num + 1, obj_set_2_num)

    def test_init(self):
        ob = ObjectiveSet(self.objective_collection, self.data_collection2, self.data_collection2.states)

    def test_get_objective_names(self):
        obset = ObjectiveSet(self.objective_collection, self.data_collection2, self.data_collection2.states)
        ob_names = obset.get_objective_names()
        goals = [self.objective.name, self.curve_objective.name]

        for goal_name, ob_name in zip(goals, ob_names):
            self.assertTrue(goal_name in ob_name)
        
    def test_get_objective_states(self):
        obset = ObjectiveSet(self.objective_collection, self.data_collection2, self.data_collection2.states)
        ob_set_states = obset.states
        
        for state_name in ob_set_states.keys():
            self.assertEqual(self.data_collection2.states[state_name], ob_set_states[state_name])

    def test_get_objective_data_collection(self):
        obset = ObjectiveSet(self.objective_collection, self.data_collection2, self.data_collection2.states)
        ob_set_dc = obset.data_collection
        
        for state_name in ob_set_dc.keys():
            self.assertTrue(np.array_equal(np.asarray(self.data_collection2[state_name]), np.asarray(ob_set_dc[state_name])))

    def test_bad_init(self):
        with self.assertRaises(ObjectiveSet.TypeError):
            ob = ObjectiveSet("x", None, None)
        with self.assertRaises(ObjectiveSet.TypeError):
            ob = ObjectiveSet(self.objective_collection, None, self.state_collection)
        with self.assertRaises(ObjectiveSet.TypeError):
            ob = ObjectiveSet("x", self.data_collection, self.state_collection)
        with self.assertRaises(ObjectiveSet.TypeError):
            ob = ObjectiveSet(self.objective_collection, self.data_collection, None)

    def test_get_num_residuals_and_objectives(self):
        obj_set = ObjectiveSet(self.objective_collection, self.data_collection2, self.data_collection2.states)

        num_resids = obj_set.residual_vector_length
        num_objectives = obj_set.number_of_objectives

        self.assertEqual(num_resids, 10 * 2)
        self.assertEqual(num_objectives, 2)

    def test_get_num_residuals_and_objectives_repeats(self):
        self.data_collection2.add(self.data_2_state)
        self.data_collection2.add(self.data_3_state)

        obj_set = ObjectiveSet(self.objective_collection, self.data_collection2, self.data_collection2.states)

        num_resids = obj_set.residual_vector_length
        num_objectives = obj_set.number_of_objectives

        self.assertEqual(num_resids, 10 * 2+10+10)
        self.assertEqual(num_objectives, 2)

    def test_get_num_residuals_and_objectives_repeats_different_length(self):
        self.data_collection2.add(self.data_2_state)
        self.data_collection2.add(self.data_3_state)
        self.data_collection2.add(self.data_single_row_ones_state)

        obj_set = ObjectiveSet(self.objective_collection, self.data_collection2, self.data_collection2.states)

        num_resids = obj_set.residual_vector_length
        num_objectives = obj_set.number_of_objectives

        self.assertEqual(num_resids, 10 * 2+10+10+2*1)
        self.assertEqual(num_objectives, 2)

    def test_get_num_residuals_and_objectives_repeats_different_length_max_extractor_objective(self):
        self.data_collection2.add(self.data_2_state)
        self.data_collection2.add(self.data_3_state)
        self.data_collection2.add(self.data_single_row_ones_state)

        objective = Objective("Y")
        objective.set_qoi_extractors(MaxExtractor("Y"))
        obj_col = ObjectiveCollection("test", objective)
        obj_set = ObjectiveSet(obj_col, self.data_collection2, self.data_collection2.states)

        num_resids = obj_set.residual_vector_length
        num_objectives = obj_set.number_of_objectives

        self.assertEqual(num_resids, 5)
        self.assertEqual(num_objectives, 1)

    def test_get_num_residuals_repeats_different_length_max_extractor_objective_with_data_purge(self):
        self.data_collection2.add(self.data_2_state)
        self.data_collection2.add(self.data_3_state)
        self.data_collection2.add(self.data_single_row_ones_state)

        objective = Objective("Y")
        objective.set_qoi_extractors(MaxExtractor("Y"))
        objective.set_name('objective')
        obj_col = ObjectiveCollection("test", objective)
        obj_set = ObjectiveSet(obj_col, self.data_collection2, self.data_collection2.states)
        obj_set.purge_unused_data()
        sim_data_dict  = {"X":np.linspace(0,1,5), "Y":np.linspace(0,1,5)}
        sim_data_state = convert_dictionary_to_data(sim_data_dict)
        sim_data_state.set_state(self.state)
        sim_data_state_2 = convert_dictionary_to_data(sim_data_dict)
        sim_data_state_2.set_state(self.state_2)

        sim_dc = DataCollection('col', sim_data_state, sim_data_state_2)

        res, qois = obj_set.calculate_objective_set_results(sim_dc)
        res = res["objective"]
        self.assertEqual(len(res.get_flattened_weighted_conditioned_residuals()), 5)

    def test_get_conditioned_data(self):

        obj_set = ObjectiveSet(self.objective_collection, self.data_collection2, self.data_collection2.states)

        c_data_set = obj_set.conditioned_experiment_qoi_collection
        for objective in self.objective_collection.values():
            self.assertIn(self.state.name, c_data_set[objective.name].state_names)
            self.assertIn(self.state_2.name, c_data_set[objective.name].state_names)

            goal1 = (self.data_mat[:, 1] + 1) / 3.0
            goal2 = (self.data_mat_2[:, 1] + 1.5) / 4.5
            d1 = c_data_set[objective.name][self.state][0]["Y"]
            d2 = c_data_set[objective.name][self.state_2][0]["Y"]

            self.assertTrue(np.allclose(d1, goal1))
            self.assertTrue(np.allclose(d2, goal2))

    def test_get_flattened_weighted_conditioned_qois(self):

        dc = DataCollection("test dc", self.data_state, self.data_2_state, self.data_3_state, self.data_2_state_2)
        max_objective = Objective("Y")
        max_objective.set_qoi_extractors(MaxExtractor("X"))
        self.objective_collection.add(max_objective)
        obj_set = ObjectiveSet(self.objective_collection, dc, dc.states)

        c_data_set = obj_set.conditioned_experiment_qoi_collection
        flattened_c_data = obj_set.get_flattened_weighted_conditioned_experiment_qois()
        goal = np.array([])
        max_data = np.array([])
        for objective in self.objective_collection.values():
            for state in c_data_set[objective.name].states.values():
                for data in c_data_set[objective.name][state]:
                    for field in objective.fields_of_interest:
                        goal = np.append(goal, data[field] )
                    
        self.assertTrue(np.allclose(goal, flattened_c_data))

    def test_only_use_state_collection_subset(self):
        obj_set = ObjectiveSet(self.objective_collection, self.data_collection2, self.state_collection)

        c_data_set = obj_set.conditioned_experiment_qoi_collection
        obj_set_results, obj_set_qois = obj_set.calculate_objective_set_results(self.data_collection2)

        for objective in self.objective_collection.values():
            self.assertTrue(self.state_2 not in c_data_set[objective.name].states)

        for obj_results in obj_set_results.values():
            self.assertTrue(self.state_2 not in obj_results.objectives.keys())
            self.assertTrue(self.state_2 not in obj_results.residuals.keys())

    def test_update_objective_collection(self):
        objective_collection = ObjectiveCollection("objs", self.objective, self.curve_objective)

        obj_set = ObjectiveSet(self.objective_collection, self.data_collection2, self.state_collection)

        obj_set.update_objective_collection(objective_collection)

        self.assertEqual(obj_set.objectives, objective_collection)

    def test_user_named_objectives(self):
        self.objective.set_name("objective username")
        self.curve_objective.set_name("curve objective username")

        obj_col = ObjectiveCollection("objs", self.objective, self.curve_objective)

        obj_set = ObjectiveSet(obj_col, self.data_collection2, self.state_collection)

        names = obj_set.get_objective_names()

        goal_names = ["objective username", "curve objective username"]

        for goal_name, name in zip(goal_names, names):
            self.assertEqual(goal_name, name)


class TestObjectiveCollection(MatcalUnitTest):
    def setUp(self):
        super().setUp(__file__)

    def test_init(self):
        ob = Objective("X", "Y")
        ob2 = Objective("X")

        obj_c= ObjectiveCollection("my objs", ob, ob2)

    def test_equals(self):
        ob = Objective("X", "Y")
        ob2 = Objective("X")

        obj_c = ObjectiveCollection("my objs", ob, ob2)
        obj_c2 = copy(obj_c)
        obj_c3 = deepcopy(obj_c)

        self.assertTrue(obj_c == obj_c2)
        self.assertFalse(obj_c == obj_c3)