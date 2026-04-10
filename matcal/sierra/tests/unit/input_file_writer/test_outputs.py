from matcal.core.constants import TEMPERATURE_KEY
from matcal.core.tests.MatcalUnitTest import MatcalUnitTest

from matcal.sierra.input_file_writer import (
    SolidMechanicsUserOutput,
    SolidMechanicsUserVariable,
)
from matcal.sierra.input_file_writer.outputs import (
    SolidMechanicsNonlocalDamageAverage,
    SolidMechanicsResultsOutput,
    SolidMechanicsHeartbeatOutput,
    SolidMechanicsAdaptiveTimeStepping,
    SolidMechanicsSolutionTermination,
)


class TestUserOutputBlocks(MatcalUnitTest):
    def setUp(self):
        super().setUp(__file__)

    def test_solid_mechanics_user_output_block_nodal_field(self):
        input_block = SolidMechanicsUserOutput("load_disp_output", "grip_ns", "node set")

        input_block.add_compute_global_from_nodal_field(
            "load", "external_force(y)", calculation="sum"
        )
        input_block.add_compute_global_from_nodal_field(
            "displacement", "displacement(y)", calculation="sum"
        )
        test_strs = input_block.get_string().split("\n")
        self.assertTrue("node set = grip_ns" in test_strs[1].strip())
        self.assertTrue("compute at every step" in test_strs[2].strip())
        self.assertTrue(
            ("compute global load as sum of nodal " + "external_force(y)")
            in test_strs[3].strip()
        )
        self.assertTrue(
            ("compute global displacement as sum of " + "nodal displacement(y)")
            in test_strs[4].strip()
        )

    def test_solid_mechanics_user_output_block_element_field(self):
        input_block = SolidMechanicsUserOutput("temps_output", "necking_section", "block")
        input_block.add_compute_global_from_element_field(
            "high_temp", TEMPERATURE_KEY, calculation="max"
        )
        input_block.add_compute_global_from_element_field(
            "avg_temp", TEMPERATURE_KEY, calculation="average"
        )
        test_strs = input_block.get_string().split("\n")
        self.assertTrue("block = necking_section" in test_strs[1].strip())
        self.assertTrue("compute at every step" in test_strs[2].strip())
        self.assertTrue(
            ("compute global high_temp as max of element " + TEMPERATURE_KEY)
            in test_strs[3].strip()
        )
        self.assertTrue(
            ("compute global avg_temp as average of " + "element temperature")
            in test_strs[4].strip()
        )

    def test_solid_mechanics_user_output_block_expression(self):
        input_block = SolidMechanicsUserOutput("load_disp_output", "grip_ns", "node set")
        input_block.add_compute_global_from_expression(
            "displacement", "partial_displacement*2;"
        )
        input_block.add_compute_global_from_expression("load", "partial_load*4;")
        test_strs = input_block.get_string().split("\n")
        self.assertTrue("node set = grip_ns" in test_strs[1].strip())
        self.assertTrue("compute at every step" in test_strs[2].strip())

        self.assertTrue(
            ('compute global displacement from expression " partial_displacement*2; "')
            in test_strs[3].strip()
        )
        self.assertTrue(
            ('compute global load from expression " partial_load*4; "')
            in test_strs[4].strip()
        )

    def test_solid_mechanics_user_output_block_element_function(self):
        input_block = SolidMechanicsUserOutput("element_outputs", "include all blocks")
        input_block.add_compute_element_as_function("test", "test_function")
        test_strs = input_block.get_string().split("\n")
        element_test = input_block.get_line_value("element test", -1)
        self.assertEqual(element_test, "test_function")
        self.assertEqual(test_strs[1].strip(), "include all blocks")

    def test_solid_mechanics_user_output_block_global_function(self):
        input_block = SolidMechanicsUserOutput("global_function", "include all blocks")
        input_block.add_compute_global_as_function("test", "test_function")
        test_strs = input_block.get_string().split("\n")
        global_test = input_block.get_line_value("global test", -1)
        self.assertEqual(global_test, "test_function")
        self.assertEqual(test_strs[1].strip(), "include all blocks")

    def test_solid_mechanics_user_output_block_element_from_element(self):
        input_block = SolidMechanicsUserOutput("element_outputs", "include all blocks")
        input_block.add_compute_element_from_element("test_avg", "test")
        test_strs = input_block.get_string().split("\n")
        self.assertEqual(
            test_strs[3].strip(),
            "compute element test_avg as volume weighted average of element test",
        )

    def test_solid_mechanics_user_output_block_add_nodal_variable_transformation(self):
        input_block = SolidMechanicsUserOutput("transform", "include all blocks")
        input_block.add_nodal_variable_transformation(
            "disp", "cyl_disp", "cyl_coord_sys"
        )

        self.assertEqual(
            input_block.lines["cyl_disp"].get_string().strip(),
            "transform nodal variable disp to coordinate system cyl_coord_sys as cyl_disp",
        )


class TestUserVariableBlocks(MatcalUnitTest):
    def setUp(self):
        super().setUp(__file__)

    def test_solid_mechanics_user_variable(self):
        input_block = SolidMechanicsUserVariable(
            "damage_increment", "element", "real", 1e-4, 1e-4, 1e-4, 1e-4
        )
        test_strs = input_block.get_string().splitlines()
        self.assertEqual("type = element real length = 4", test_strs[1].strip())
        initial_values = input_block.get_line("initial value").get_values()
        self.assertEqual(initial_values[1:], [1e-4] * 4)

    def test_solid_mechanics_user_variable_add_blocks(self):
        input_block = SolidMechanicsUserVariable(
            "damage_increment", "element", "real", 1e-4, 1e-4, 1e-4, 1e-4
        )
        input_block.add_blocks("block1", "block2")
        test_str = input_block.get_string()
        self.assertIn("block = block1 block2", test_str)


class TestOutputBlocks(MatcalUnitTest):
    def setUp(self):
        super().setUp(__file__)

    def test_solid_mechanics_nonlocal_damage_average_block(self):
        input_block = SolidMechanicsNonlocalDamageAverage(0.01)
        test_strs = input_block.get_string().splitlines()
        self.assertEqual(
            "source variable = element damage_increment", test_strs[1].strip()
        )
        self.assertEqual(
            "target_variable = element nonlocal_damage_increment", test_strs[2].strip()
        )
        self.assertEqual("radius = 0.01", test_strs[3].strip())
        self.assertEqual("distance algorithm = euclidean_graph", test_strs[4].strip())

    def test_solid_mechanics_user_output_block(self):
        input_block = SolidMechanicsResultsOutput(20)
        input_block.add_element_output("eqps_avg")
        input_block.add_element_output("test", "test_out")

        input_block.add_global_output("load")
        input_block.add_global_output("displacement")

        input_block.add_nodal_output("displacement", "displ")
        test_str = input_block.get_string()
        output_name = input_block.get_line_value("database name", -1)
        self.assertEqual(output_name, "./results/results.e")
        eqps = input_block.get_line_value("element eqps_avg", -1)
        self.assertEqual(eqps, "eqps_avg")
        self.assertTrue(input_block.has_element_output("eqps_avg"))
        self.assertTrue(input_block.has_element_output("test", "test_out"))

        displ_line_name = input_block._get_nodal_variable_line_name(
            "displacement", "displ"
        )
        displ = input_block.get_line_value(displ_line_name, -1)
        self.assertEqual(displ, "displ")
        load = input_block.get_line_value("global load", -1)
        self.assertEqual(load, "load")
        displacement = input_block.get_line_value("global displacement", -1)
        self.assertEqual(displacement, "displacement")

    def test_solid_mechanics_user_output_block_exposed_surf(self):
        input_block = SolidMechanicsResultsOutput(20)
        input_block.add_element_output("eqps_avg")
        input_block.add_include_surface("DIC_surf")
        input_block.add_exclude_blocks("necking_block")
        input_block.set_output_exposed_surface()
        test_str = input_block.get_string()
        includes = input_block.get_line_value("include", -1)
        self.assertEqual(includes, "DIC_surf")
        excludes = input_block.get_line_value("exclude", -1)
        self.assertEqual(excludes, "necking_block")
        output_mesh = input_block.get_line_value("output mesh", -1)
        self.assertEqual(output_mesh, "exposed surface")
        input_block.set_output_exposed_surface(False)
        self.assertNotIn("output mesh", input_block.lines)

    def test_solid_mechanics_heartbeat_output(self):
        input_block = SolidMechanicsHeartbeatOutput(1, "load", "displacement")
        test_str = input_block.get_string()
        timestamp = input_block.get_line_value("timestamp", -1)
        self.assertEqual(timestamp, "''")
        timestamp_format = input_block.get_line_value("timestamp", -2)
        self.assertEqual(timestamp_format, "format")
        self.assertIn("global load", input_block.lines)
        self.assertIn("global displacement", input_block.lines)

    def test_solid_mechanics_heartbeat_output_get_global_output(self):
        input_block = SolidMechanicsHeartbeatOutput(1, "load", "displacement")
        g_outputs = input_block.get_global_outputs()
        g_output_names = []
        for g_output in g_outputs:
            g_output_names.append(g_output.name)
        self.assertTrue(len(g_outputs), 2)
        self.assertIn("global load", g_output_names)
        self.assertIn("global displacement", g_output_names)

    def test_solid_mechanics_adaptive_time_stepping(self):
        input_block = SolidMechanicsAdaptiveTimeStepping()
        min_mult = input_block.get_line_value("minimum multiplier")
        self.assertEqual(min_mult, 1e-8)
        max_mult = input_block.get_line_value("maximum multiplier")
        self.assertEqual(max_mult, 1)
        self.assertEqual(len(input_block.lines), 3)
        input_block.set_cutback_factor(0.75)
        cutback = input_block.get_line_value("cutback factor")
        self.assertEqual(cutback, 0.75)
        input_block.set_growth_factor(1.25)
        growth = input_block.get_line_value("growth factor")
        self.assertEqual(growth, 1.25)
        self.assertEqual(len(input_block.lines), 5)
        input_block.set_iteration_target()
        target_its = input_block.get_line_value("target iterations")
        self.assertEqual(target_its, 75)
        window = input_block.get_line_value("iteration window")
        self.assertEqual(window, 5)
        self.assertEqual(len(input_block.lines), 7)
        input_block.set_adaptive_time_stepping_method("solver_average")
        method = input_block.get_line_value("method")
        self.assertEqual(method, "solver_average")

    def test_solution_termination(self):
        input_block = SolidMechanicsSolutionTermination()
        input_block.add_global_termination_criteria("test", 1)
        self.assertEqual(input_block.get_line_value("global test", 1), "global")
        self.assertEqual(input_block.get_line_value("global test", 2), "test")
        self.assertEqual(input_block.get_line_value("global test", 3), "<")
        self.assertEqual(input_block.get_line_value("global test", 4), 1)
        self.assertEqual(input_block.get_line_value("terminate type"), "entire_run")
