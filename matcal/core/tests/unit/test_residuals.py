from matcal.core.state import State
import numpy as np

from matcal.core.data import convert_dictionary_to_data
from matcal.core.tests.MatcalUnitTest import MatcalUnitTest
from matcal.core.residuals import LogResidualCalculator, NoiseWeightingConstant,\
    NominalResidualCalculator, get_array, IdentityWeighting, \
    LinearDataSizeNormalizer, SqrtDataSizeNormalizer, UserFunctionWeighting, \
    NoiseWeightingFromFile, ResidualWeightingBase, ConstantFactorWeighting, \
    InvertedLinearDataSizeNormalizer, InvertedSqrtDataSizeNormalizer


class TestResiduals:
    def __init__():
        pass
    class CommonTests(MatcalUnitTest):
        def test_raise_invalid_field_types(self):
            with self.assertRaises(NominalResidualCalculator.FieldNameTypeError):
                self.RC("F", None)
            with self.assertRaises(NominalResidualCalculator.FieldNameTypeError):
                self.RC(None, "D")
            with self.assertRaises(NominalResidualCalculator.FieldNameTypeError):
                self.RC("I", "D", None)
            with self.assertRaises(NominalResidualCalculator.FieldNameTypeError):
                self.RC(1, "D")
            with self.assertRaises(NominalResidualCalculator.FieldsOfInterestNotDefined):
                self.RC()


class TestNominalResiduals(TestResiduals.CommonTests):
    RC = NominalResidualCalculator

    def setUp(self):
        super().setUp(__file__)

    def _get_residual_base(self, ref_dict, eval_dict, *dep_fields):
        residual = self.RC(*dep_fields)
        ref_data = convert_dictionary_to_data(ref_dict)
        eval_data = convert_dictionary_to_data(eval_dict)

        return residual.calculate(ref_data, eval_data)

    def test_calculate_zero(self):
        example_dict = {"I": [0, 1, 2, 3, 4], "D": [0, 2, 4, 6, 8]}
        goal = np.zeros(5)
        r = self._get_residual_base(example_dict, example_dict,  "D")
        
        self.assertAlmostEqual(np.linalg.norm(r["D"] - goal), 0)
        

    def test_calculate_two_zero_sets(self):
        example_dict = {"I": [0, 1, 2, 3, 4], "D": [0, 2, 4, 6, 8], "DD": [-0, -2, -4, -6, -8]}
        goal = np.zeros(5)
        r = self._get_residual_base(example_dict, example_dict, "D", "DD")
        self.assertAlmostEqual(np.linalg.norm(r["D"] - goal), 0)
        self.assertAlmostEqual(np.linalg.norm(r["DD"] - goal), 0)

    def test_calculate_data_missing_fields(self):
        example_dict = {"I": [0, 1, 2, 3, 4], "D": [0, 2, 4, 6, 8], "DD": [-0, -2, -4, -6, -8]}
        example_dict2 = {"B": [0, 1, 2, 3, 4], "C": [0, 2, 4, 6, 8], "E": [-0, -2, -4, -6, -8]}
        
        goal = np.zeros(5)
        
        with self.assertRaises(NominalResidualCalculator.DataMissingFieldsOfInterest):
            r = self._get_residual_base(example_dict, example_dict, "A")
        with self.assertRaises(NominalResidualCalculator.DataMissingFieldsOfInterest):
            r = self._get_residual_base(example_dict, example_dict2, "D")
        
    def test_calculate_ones(self):
        ref_dict = {"I": [0, 1, 2, 3, 4], "D": [0, 2, 4, 6, 8]}
        eval_dict = {"I": [0, 1, 2, 3, 4], "D": [1, 3, 5, 7, 9]}
        goal = np.ones(5)
        r = self._get_residual_base(ref_dict, eval_dict, "D")
        self.assertAlmostEqual(np.linalg.norm(r["D"] - goal), 0)

class TestLogResiduals(TestResiduals.CommonTests):
    RC = LogResidualCalculator

    def setUp(self):
        super().setUp(__file__)

    def _get_residual_base(self, ref_dict, eval_dict, *dep_fields):
        residual = self.RC(*dep_fields)
        ref_data = convert_dictionary_to_data(ref_dict)
        eval_data = convert_dictionary_to_data(eval_dict)

        return residual.calculate(ref_data, eval_data)

    def test_calculate_zero(self):
        example_dict = {"I": [0, 1, 2, 3, 4], "D": [1, 2, 4, 6, 8]}
        goal = np.zeros(5)
        r = self._get_residual_base(example_dict, example_dict,  "D")
        self.assert_close_arrays(r['D'], goal)
        

    def test_calculate_two_zero_sets(self):
        example_dict = {"I": [0, 1, 2, 3, 4], "D": [1, 2, 4, 6, 8], "DD": [10, 20, 40, 60, 80]}
        goal = np.zeros(5)
        r = self._get_residual_base(example_dict, example_dict, "D", "DD")
        self.assert_close_arrays(r['D'], goal)
        self.assert_close_arrays(r['DD'], goal)


    def test_calculate_data_missing_fields(self):
        example_dict = {"I": [1, 1, 2, 3, 4], "D": [1, 2, 4, 6, 8], "DD": [-1, -2, -4, -6, -8]}
        example_dict2 = {"B": [1, 1, 2, 3, 4], "C": [1, 2, 4, 6, 8], "E": [-1, -2, -4, -6, -8]}
        
        goal = np.zeros(5)
        
        with self.assertRaises(NominalResidualCalculator.DataMissingFieldsOfInterest):
            r = self._get_residual_base(example_dict, example_dict, "A")
        with self.assertRaises(NominalResidualCalculator.DataMissingFieldsOfInterest):
            r = self._get_residual_base(example_dict, example_dict2, "D")
        
    def test_calculate_ones(self):
        ref_dict = {"I": [1, 1, 2, 3, 4], "D": [1, 2, 4, 6, 8]}
        goal = np.ones(5)
        new_D = np.exp(np.log(ref_dict['D'])+1)
        eval_dict = {"I": [1, 1, 2, 3, 4], "D": new_D}
        
        r = self._get_residual_base(ref_dict, eval_dict, "D")
        self.assert_close_arrays(r['D'], goal, show_on_fail=True)




class TestResidualWithVFMData(MatcalUnitTest):

    def setUp(self):
        super().setUp(__file__)

    def test_refactor_outline(self):
        rc = NominalResidualCalculator('virtual_power')
        external_power = convert_dictionary_to_data({'virtual_power': np.zeros(3)})
        internal_power = convert_dictionary_to_data(({'virtual_power': np.zeros(3)}))
        residual = rc.calculate(external_power, internal_power)
        self.assert_close_arrays(get_array(residual),  np.zeros(3))


class DataSizeNormalizer:
    def __init__():
        pass

    class CommonTests(MatcalUnitTest):       

        def test_apply_return_same(self):
            residual = np.array([1])
            weighted_residual = self.DSW.apply(residual)
            self.assertEqual(residual, weighted_residual)

        def test_apply_returned_halved(self):
            residual = np.array([1., 2])
            weighted_residual = self.DSW.apply( residual)
            self.assert_close_arrays(residual / self._n_correction(2), weighted_residual)

        def test_random_n_length(self):
            n = 16
            n_scale = self._n_correction(n)
            residual = np.random.random(n)
            weighted_residual = self.DSW.apply(residual)
            goal = residual / n_scale
            self.assert_close_arrays(goal, weighted_residual)

        def test_multiple_fields(self):
            n = 20
            n_scale = self._n_correction(2*n)
            a = np.random.random(n)
            b = np.random.random(n)
            residual = np.concatenate([a, b])
            weighted_residual = self.DSW.apply(residual)
            self.assert_close_arrays(residual / n_scale, weighted_residual)
        

class TestSqrtSizeNormalizer(DataSizeNormalizer.CommonTests):

    def setUp(self):
        super().setUp(__file__)
        self.DSW = SqrtDataSizeNormalizer()
        self._n_correction = np.sqrt


class TestLinearSizeNormalizer(DataSizeNormalizer.CommonTests):

    def identity(self, a):
        return a

    def setUp(self):
        super().setUp(__file__)
        self.DSW = LinearDataSizeNormalizer()
        self._n_correction = self.identity 
    

class TestInvertedLinearSizeNormalizer(DataSizeNormalizer.CommonTests):


    def setUp(self):
        super().setUp(__file__)
        self.DSW = InvertedLinearDataSizeNormalizer()
        self._n_correction = self.inverted_linear_correction

    def inverted_linear_correction(self, n):
        return 1.0/n

class TestInvertedSqrtSizeNormalizer(DataSizeNormalizer.CommonTests):

    def setUp(self):
        super().setUp(__file__)
        self.DSW = InvertedSqrtDataSizeNormalizer()
        self._n_correction = self.inverted_sqrt_correction

    def inverted_sqrt_correction(self, n):
        return 1.0/np.sqrt(n)

class TestConstantFactorWeighting(MatcalUnitTest):

    def setUp(self):
        super().setUp(__file__)
        self.DSW = ConstantFactorWeighting(4)

    def test_bad_init(self):
        with self.assertRaises(ResidualWeightingBase.TypeError):
            weighting = ConstantFactorWeighting("A")

    def test_apply_return_same(self):
        residual = {"F1": np.array([1])}
        weighted_residual = self.DSW.apply(None, None, convert_dictionary_to_data(residual))
        self.assertEqual(residual['F1'][0] * 4, weighted_residual['F1'][0])

    def test_apply_returned_halved(self):
        a = np.array([1., 2])
        residual = {"F1": a}
        weighted_residual = self.DSW.apply(None, None, convert_dictionary_to_data(residual))
        self.assert_close_arrays(residual['F1'] * 4, weighted_residual['F1'])

    def test_random_n_length(self):
        n = 20
        a = np.random.random(n)
        residual = {"F": a}
        weighted_residual = self.DSW.apply(None, None, convert_dictionary_to_data(residual))
        self.assert_close_arrays(a * 4, weighted_residual["F"])

    def test_multiple_fields(self):
        n = 20
        a = np.random.random(n)
        b = np.random.random(n)
        residual = {"F": a, "A": b}
        weighted_residual = self.DSW.apply(None, None, convert_dictionary_to_data(residual))
        self.assert_close_arrays(a * 4, weighted_residual["F"])
        self.assert_close_arrays(b * 4, weighted_residual["A"])


class TestIdentityWeighting(MatcalUnitTest):

    def setUp(self):
        super().setUp(__file__)
    
    def test_get_same_residual(self):
        n = 20
        a = np.random.random(n)
        b = np.random.random(n)
        names = ['indep', "target"]
        residual = {"indep": a, "target": b}
        weighted_residual = IdentityWeighting().apply(None, None, convert_dictionary_to_data(residual))
        for name in names:
            self.assert_close_arrays(residual[name], weighted_residual[name])


class TestUserFunctionWeighting(MatcalUnitTest):

    def setUp(self):
        super().setUp(__file__)

    def test_lambda_doubling(self):
        fun = lambda x, y, res: 2 * res
        UFW = UserFunctionWeighting('indep', 'target', fun)

        n = 20
        sample_data = {"indep": np.linspace(0, 10, n), "target": np.linspace(0, 15, n)}
        a = np.random.random(n)
        b = np.random.random(n)
        residual = {"indep": a, "target": b}
        weighted_residual = UFW.apply(sample_data, sample_data, convert_dictionary_to_data(residual))
        self.assert_close_arrays(weighted_residual['target'], residual['target'] * 2)

    def test_def_function(self):
        def fun(independent_field, target_field, residual):
            import numpy as np
            return np.multiply(independent_field, residual)

        UFW = UserFunctionWeighting('indep', 'target', fun)

        n = 20
        d = np.linspace(0, 10, n)
        sample_data = {"indep": d, "target": d}
        a = np.random.random(n)
        b = np.random.random(n)
        residual = {"indep": a, "target": b}
        weighted_residual = UFW.apply(sample_data, sample_data, convert_dictionary_to_data(residual))
        self.assert_close_arrays(weighted_residual['target'], np.multiply(residual['target'], d))


class TestNoiseWeightingFromFile(MatcalUnitTest):

    def setUp(self):
        super().setUp(__file__)

    def test_init(self):
        nwff = NoiseWeightingFromFile('field_name')

    def test_uniform_weighting(self):
        n = 20
        scale = 10
        a = np.random.random(n)
        sample_data = {"F": a, "F_noise": np.ones(n) * scale}
        nwff = NoiseWeightingFromFile('F')

        weighted_residual = nwff.apply(sample_data, sample_data, convert_dictionary_to_data(sample_data))
        self.assert_close_arrays(a / scale, weighted_residual["F"])

    def test_unique_weighting(self):
        n = 20
        scale = 10
        a = np.random.random(n)
        sample_data = {"F": a, "F_noise": a}
        nwff = NoiseWeightingFromFile('F')

        weighted_residual = nwff.apply(sample_data, sample_data, convert_dictionary_to_data(sample_data))
        self.assert_close_arrays(np.ones(n), weighted_residual["F"])


class TestNoiseWeightingConstant(MatcalUnitTest):
    
    def setUp(self):
        super().setUp(__file__)
        
    def test_init(self):
        field_noise = {'a':.1, 'b':2}
        nwc = NoiseWeightingConstant(**field_noise)
        
    def test_scale_residual_indep_of_reference_data(self):
        n_points = 20
        noise_b = 2
        noise_a = .1
        nwc = NoiseWeightingConstant(a=noise_a, b=noise_b)
        res_dict = {'a':np.linspace(0, 1, n_points), 'b':np.linspace(50, 80, n_points)}
        fake_resid = convert_dictionary_to_data(res_dict)
        weighted_resid = nwc.apply(None, None, fake_resid)
        self.assert_close_arrays(fake_resid['a'] / noise_a, weighted_resid['a'])
        self.assert_close_arrays(fake_resid['b'] / noise_b, weighted_resid['b'])     
        
    def test_scale_residual_indep_of_reference_data_two_states(self):
        n_points = 20
        noise_b = 2
        noise_a = .1
        nwc = NoiseWeightingConstant(a=noise_a, b=noise_b)
        fake_resid1 = convert_dictionary_to_data({'a':np.linspace(0, 1, n_points), 'b':np.linspace(50, 80, n_points)})
        fake_resid2 = convert_dictionary_to_data({'a':np.linspace(2, 4, n_points), 'b':np.linspace(-50, 50, n_points)})
        s1 = State("1")
        s2 = State("2")
        
        fake_resid1.set_state(s1)
        fake_resid2.set_state(s2)
        for cur_res in [fake_resid1, fake_resid2]:
            weighted_resid = nwc.apply(None, None, cur_res)
            self.assert_close_arrays(cur_res['a'] / noise_a, weighted_resid['a'])
            self.assert_close_arrays(cur_res['b'] / noise_b, weighted_resid['b'])
            
    def test_raise_error_if_over_defined(self):
        n_points = 20
        noise_b = 2
        noise_a = .1
        state_weights = {}
        state_weights['1'] = {'a':noise_a, 'b':noise_b}
        state_weights['2'] = {'a':2*noise_a, 'b':2*noise_b}
        with self.assertRaises(RuntimeError):
            nwc = NoiseWeightingConstant(state_weights, a=noise_a, b=noise_b)
    
    def test_raise_error_if_mixed_defintions(self):
        n_points = 20
        noise_b = 2
        noise_a = .1
        state_weights = {}
        state_weights['1'] = {'a':noise_a, 'b':noise_b}
        state_weights['c'] = 100
        with self.assertRaises(ValueError):
            nwc = NoiseWeightingConstant(state_weights)
    
    
    
    def test_scale_residual_diff_states_data_two_states(self):
        n_points = 20
        noise_b = 2
        noise_a = .1
        state_weights = {}
        state_weights['1'] = {'a':noise_a, 'b':noise_b}
        state_weights['2'] = {'a':2*noise_a, 'b':2*noise_b}
        nwc = NoiseWeightingConstant(state_weights)
        fake_resid1 = convert_dictionary_to_data({'a':np.linspace(0, 1, n_points), 'b':np.linspace(50, 80, n_points)})
        fake_resid2 = convert_dictionary_to_data({'a':np.linspace(2, 4, n_points), 'b':np.linspace(-50, 50, n_points)})
        s1 = State("1")
        s2 = State("2")
        
        fake_resid1.set_state(s1)
        fake_resid2.set_state(s2)
        for scale, cur_res in enumerate([fake_resid1, fake_resid2]):
            weighted_resid = nwc.apply(None, None, cur_res)
            self.assert_close_arrays(cur_res['a'] / ((scale+1) * noise_a), weighted_resid['a'])
            self.assert_close_arrays(cur_res['b'] / ((scale +1)* noise_b), weighted_resid['b'])