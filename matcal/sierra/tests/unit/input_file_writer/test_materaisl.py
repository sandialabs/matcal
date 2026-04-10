from matcal.core.tests.MatcalUnitTest import MatcalUnitTest
from matcal.sierra.input_file_writer.materials import ThermalMaterial


class TestThermalMaterial(MatcalUnitTest):
    def setUp(self):
        super().setUp(__file__)

    def test_aria_material_(self):
        mat = ThermalMaterial(1, 2, 3)
        test_str = mat.get_string()
        self.assertTrue("Begin aria material matcal_thermal" in test_str)
        self.assertTrue("density = constant rho = 1" in test_str)
        self.assertTrue("thermal conductivity = constant K = 2" in test_str)
        self.assertTrue("specific heat = constant cp = 3" in test_str)
        self.assertTrue("heat conduction = basic" in test_str)
