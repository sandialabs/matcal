from matcal.core.tests.MatcalUnitTest import MatcalUnitTest

from matcal.sierra.input_file_writer.boundary_conditions import (
    SolidMechanicsFixedDisplacement,
    SolidMechanicsPrescribedDisplacement,
    SolidMechanicsPrescribedTemperature,
    SolidMechanicsInitialTemperature,
)


class TestSolidMechanicsBoundaryConditions(MatcalUnitTest):
    def setUp(self):
        super().setUp(__file__)

    def test_solid_mechanics_fixed_disp_block(self):
        input_block = SolidMechanicsFixedDisplacement("fixed_x_ns", "x")
        test_str = input_block.get_string()
        self.assertEqual("Begin fixed displacement", test_str.split("\n")[0])
        node_set_values = input_block.lines["node set"].get_values()
        self.assertEqual("fixed_x_ns", node_set_values[-1])
        dir_values = input_block.lines["component"].get_values()
        self.assertEqual("x", dir_values[-1])

    def test_solid_mechanics_prescribed_disp_block(self):
        input_block = SolidMechanicsPrescribedDisplacement("disp_func", "grip_ns", "x")
        function_values = input_block.lines["function"].get_values()
        self.assertEqual("disp_func", function_values[-1])
        self.assertNotIn("scale factor", input_block.lines)

        input_block = SolidMechanicsPrescribedDisplacement(
            "disp_func", "grip_ns", "x", scale_factor=0.5
        )
        scale_factor_values = input_block.lines["scale factor"].get_values()
        self.assertEqual(0.5, scale_factor_values[-1])

    def test_solid_mechanics_prescribed_temperature_block(self):
        input_block = SolidMechanicsPrescribedTemperature(
            "include all blocks", function_name="temp_func"
        )
        test_strs = input_block.get_string().splitlines()

        self.assertTrue("include all blocks" in test_strs[1].strip())
        self.assertTrue("function" in test_strs[2].strip())

        input_block = SolidMechanicsPrescribedTemperature("temp_nodes", function_name="temp_func")
        test_str = input_block.get_string()
        test_strs = test_str.splitlines()

        self.assertEqual("node set = temp_nodes", test_strs[1].strip())
        self.assertTrue("receive from transfer" not in test_str)

        input_block = SolidMechanicsPrescribedTemperature("temp_nodes", transfer=True)
        test_str = input_block.get_string()
        self.assertTrue("receive from transfer" in test_str)
        self.assertTrue("function" not in test_str)

    def test_solid_mechanics_prescribed_temperature_read_from_mesh(self):
        input_block = SolidMechanicsPrescribedTemperature("include all blocks")
        input_block.read_from_mesh("temp")
        test_strs = input_block.get_string().splitlines()
        self.assertTrue("include all blocks" in test_strs[1].strip())
        self.assertTrue("read variable = temp" in test_strs[2].strip())

    def test_solid_mechanics_initial_temperature_block(self):
        input_block = SolidMechanicsInitialTemperature("include all blocks", 20)
        test_strs = input_block.get_string().splitlines()
        self.assertEqual("include all blocks", test_strs[1].strip())
        self.assertEqual("magnitude = 20", test_strs[2].strip())

        input_block = SolidMechanicsInitialTemperature("temp_block", 20)
        test_strs = input_block.get_string().splitlines()
        self.assertEqual("block = temp_block", test_strs[1].strip())
