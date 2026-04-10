from matcal.core.tests.MatcalUnitTest import MatcalUnitTest

from matcal.sierra.input_file_writer.regions_models import (
    SolidMechanicsRegion,
    _FiniteElementModelNames,
)
from matcal.sierra.input_file_writer.time_control import (
    SolidMechanicsProcedure,
    ThermalTimeParameters,
)


class TestSolidMechanicsProcedure(MatcalUnitTest):
    def setUp(self):
        super().setUp(__file__)

    def test_adagio_proc(self):
        region_block = SolidMechanicsRegion(
            "adagio_region", _FiniteElementModelNames.solid_mechanics
        )
        input_subblock = SolidMechanicsProcedure("adagio_proc", region_block, 0, 1, 100)
        test_str = input_subblock.get_string()

        self.assertTrue("Begin adagio procedure adagio_proc\n" in test_str)
        self.assertTrue("Begin time control\n" in test_str)
        self.assertTrue("termination time =" in test_str)
        self.assertTrue("start time =" in test_str)
        self.assertTrue("time increment =" in test_str)
        self.assertTrue("time stepping block elastic_init" in test_str)
        self.assertTrue("time stepping block load" in test_str)
        self.assertTrue("Begin parameters for adagio region adagio_region" in test_str)

        tc_subblock = input_subblock.get_subblock("time control")
        elastic_init_subblock = tc_subblock.get_subblock("elastic_init")
        self.assertEqual(elastic_init_subblock.get_line_value("start time"), 0)

        load_subblock = tc_subblock.get_subblock("load")
        self.assertEqual(load_subblock.get_line_value("start time"), 0.01 * 1e-3)

        elatic_init_params = elastic_init_subblock.get_subblock("adagio_region")
        self.assertEqual(elatic_init_params.get_line_value("time increment"), 0.01 * 1e-3)

        load_params = load_subblock.get_subblock("adagio_region")
        self.assertEqual(load_params.get_line_value("time increment"), 0.01)

    def test_adagio_proc_set_number_of_time_steps(self):
        region_block = SolidMechanicsRegion(
            "adagio_region", _FiniteElementModelNames.solid_mechanics
        )
        input_subblock = SolidMechanicsProcedure("adagio_proc", region_block, 0, 1, 100)
        self.assertEqual(input_subblock._time_steps, 100)
        input_subblock.set_number_of_time_steps(1000)
        self.assertEqual(input_subblock._time_steps, 1000)

        tc_subblock = input_subblock.get_subblock("time control")
        elastic_init_subblock = tc_subblock.get_subblock("elastic_init")
        self.assertEqual(elastic_init_subblock.get_line_value("start time"), 0)

        load_subblock = tc_subblock.get_subblock("load")
        self.assertEqual(load_subblock.get_line_value("start time"), 0.001 * 1e-3)

        elatic_init_params = elastic_init_subblock.get_subblock("adagio_region")
        self.assertEqual(elatic_init_params.get_line_value("time increment"), 0.001 * 1e-3)

        load_params = load_subblock.get_subblock("adagio_region")
        self.assertEqual(load_params.get_line_value("time increment"), 0.001)

    def test_adagio_proc_set_start_time(self):
        region_block = SolidMechanicsRegion(
            "adagio_region", _FiniteElementModelNames.solid_mechanics
        )
        input_subblock = SolidMechanicsProcedure("adagio_proc", region_block, 0, 1, 100)
        self.assertEqual(input_subblock._start_time, 0)
        input_subblock.set_start_time(0.1)
        self.assertEqual(input_subblock._start_time, 0.1)

        tc_subblock = input_subblock.get_subblock("time control")
        elastic_init_subblock = tc_subblock.get_subblock("elastic_init")
        self.assertEqual(elastic_init_subblock.get_line_value("start time"), 0.1)

        ts = (1 - 0.1) / 100
        load_subblock = tc_subblock.get_subblock("load")
        self.assertEqual(load_subblock.get_line_value("start time"), 0.1 + ts * 1e-3)

        elatic_init_params = elastic_init_subblock.get_subblock("adagio_region")
        self.assertEqual(elatic_init_params.get_line_value("time increment"), ts * 1e-3)

        load_params = load_subblock.get_subblock("adagio_region")
        self.assertEqual(load_params.get_line_value("time increment"), ts)

    def test_adagio_proc_set_end_time(self):
        region_block = SolidMechanicsRegion(
            "adagio_region", _FiniteElementModelNames.solid_mechanics
        )
        input_subblock = SolidMechanicsProcedure("adagio_proc", region_block, 0, 1, 100)
        self.assertEqual(input_subblock._termination_time, 1)
        input_subblock.set_end_time(10)
        self.assertEqual(input_subblock._termination_time, 10)

        tc_subblock = input_subblock.get_subblock("time control")
        elastic_init_subblock = tc_subblock.get_subblock("elastic_init")
        self.assertEqual(elastic_init_subblock.get_line_value("start time"), 0.0)

        ts = (10 - 0.0) / 100
        load_subblock = tc_subblock.get_subblock("load")
        self.assertEqual(load_subblock.get_line_value("start time"), 0 + ts * 1e-3)

        elatic_init_params = elastic_init_subblock.get_subblock("adagio_region")
        self.assertEqual(elatic_init_params.get_line_value("time increment"), ts * 1e-3)

        load_params = load_subblock.get_subblock("adagio_region")
        self.assertEqual(load_params.get_line_value("time increment"), ts)


class TestThermalTimeParameters(MatcalUnitTest):
    def setUp(self):
        super().setUp(__file__)

    def test_aria_time_parameters(self):
        input_block = ThermalTimeParameters("my_region", 0.01)
        self.assertEqual(input_block.name, "my_region")
        self.assertEqual(input_block.lines["initial time step size"].get_values()[-1], 0.01)
        input_block.set_time_increment(1e-3)
        self.assertEqual(input_block.lines["initial time step size"].get_values()[-1], 0.001)
