from matcal.core.tests.MatcalUnitTest import MatcalUnitTest

from matcal.sierra.input_file_writer import (
    SierraGlobalDefinitions,
)
from matcal.sierra.input_file_writer.blocks_base import (
    _get_default_coupled_procedure_name,
    _get_default_thermal_region_name,
    AnalyticSierraFunction, PiecewiseLinearFunction
)


class TestSierraGlobalDefinitions(MatcalUnitTest):
    def setUp(self):
        super().setUp(__file__)

    def test_sierra_global_defs_bloc(self):
        global_defs = SierraGlobalDefinitions()
        test_str = global_defs.get_string()
        self.assertEqual(len(test_str.split("\n")), 11)
        self.assertTrue("cylindrical" in test_str)
        self.assertTrue("rectangular" in test_str)
        self.assertTrue("axis" in test_str)

class TestSierraFunctions(MatcalUnitTest):
    def setUp(self):
        super().setUp(__file__)

    def test_analytic_function_(self):
        func = AnalyticSierraFunction("double_volume")
        func.add_expression_variable("volume", "element", "volume")
        func.add_evaluation_expression("2*volume")
        test_str = func.get_string()
        self.assertTrue("Begin" in test_str)
        self.assertTrue("End" in test_str)
        self.assertTrue("function" in test_str)
        self.assertTrue("double_volume" in test_str)
        self.assertTrue("2*volume" in test_str)
        self.assertTrue("expression variable" in test_str)
        self.assertTrue("evaluate expression" in test_str)
        self.assertTrue("type is analytic" in test_str)
        self.assertTrue("volume = element volume" in test_str)
    
    def test_piecewise_linear_function_(self):
        func = PiecewiseLinearFunction("piecewise_linear", [0,1], [0, 1])
        test_str = func.get_string()
        self.assertTrue("Begin function piecewise_linear" in test_str)
        self.assertTrue("End function piecewise_linear" in test_str)
        self.assertTrue("type is piecewise linear" in test_str)
        self.assertTrue("Begin Values" in test_str)
        self.assertTrue("End Values" in test_str)
        self.assertTrue("0 0" in test_str)
        self.assertTrue("1 1" in test_str)
        
        func.scale_function(x = 2)
        test_str = func.get_string()
        self.assertTrue("x scale = 2" in test_str)
        func.scale_function(y = 3)
        test_str = func.get_string()
        self.assertTrue("y scale = 3" in test_str)
        self.assertTrue("x scale" not in test_str)

class TestBlocksBaseDefaults(MatcalUnitTest):
    def setUp(self):
        super().setUp(__file__)

    def test_default_names(self):
        self.assertIsInstance(_get_default_coupled_procedure_name(), str)
        self.assertIsInstance(_get_default_thermal_region_name(), str)
        self.assertTrue(len(_get_default_coupled_procedure_name()) > 0)
        self.assertTrue(len(_get_default_thermal_region_name()) > 0)